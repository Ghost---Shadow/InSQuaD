from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time
from checkpoint_manager import CheckpointManager
from config import RootConfig
from dataloaders import DATALOADERS_LUT
from dense_indexes import DENSE_INDEXES_LUT
from eval_utils import evaluate_with_options_if_possible, get_options_if_possible
from generative_models import GENERATIVE_MODELS_LUT
from losses import LOSSES_LUT
from prompt_formatting_strategies import PROMPT_FORMATTING_STRATEGIES_LUT
from semantic_search_models import SEMANTIC_SEARCH_MODELS_LUT
from shortlist_strategies import SHORTLIST_STRATEGIES_LUT
from subset_selection_strategies import SUBSET_SELECTION_STRATEGIES_LUT
import torch
from tqdm import tqdm
from train_utils import count_rows_jsonl, generate_artifacts_dir, set_seed


class OfflineEvaluationPipeline:
    def __init__(self, config: RootConfig):
        self.config = config
        self.num_shots = config.offline_validation.num_shots
        self.current_seed = None
        self.current_dataset_name = None
        self._load_parts(config)

    def cleanup(self):
        self.semantic_search_model.model.cpu()
        del self.semantic_search_model
        self.semantic_search_model = None
        gc.collect()

        self.generative_model.model.cpu()
        del self.generative_model
        self.generative_model = None
        gc.collect()

        # Hopefully fix oom
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def is_done(self):
        if self.current_dataset_name is None:
            return False
        if self.current_dataset_name is None:
            return False
        return os.path.exists(self.final_result_json_path)

    def _load_parts(self, config: RootConfig):
        # Generative Model
        print("Loading generative model")
        generative_model_type = config.offline_validation.generative_model.type
        generative_model_config = config.offline_validation.generative_model
        self.generative_model = GENERATIVE_MODELS_LUT[generative_model_type](
            config, generative_model_config
        )

        # Semantic Search Model
        print("Loading Semantic Search Model")
        semantic_search_model_type = config.architecture.semantic_search_model.type
        self.semantic_search_model = SEMANTIC_SEARCH_MODELS_LUT[
            semantic_search_model_type
        ](config)
        self.checkpoint_manager = CheckpointManager(self)

        # Subset Selection Strategy
        subset_selection_strategy_type = (
            config.architecture.subset_selection_strategy.type
        )
        self.subset_selection_strategy = SUBSET_SELECTION_STRATEGIES_LUT[
            subset_selection_strategy_type
        ](config, self)

        # Dense Index
        print("Loading Dense Index")
        dense_index_type = config.architecture.dense_index.type
        self.dense_index = DENSE_INDEXES_LUT[dense_index_type](config, self)

        # Prompt Formatting Strategy
        prompt_formatting_strategy_type = (
            config.architecture.prompt_formatting_strategy.type
        )
        self.prompt_formatting_strategy = PROMPT_FORMATTING_STRATEGIES_LUT[
            prompt_formatting_strategy_type
        ](config)

        # Shortlist Selection Strategy
        shortlist_strategy_type = config.offline_validation.type
        self.shortlist_strategy = SHORTLIST_STRATEGIES_LUT[shortlist_strategy_type](
            config, self
        )

        # Loss Function (for metrics and shortlisting)
        loss_function_type = config.training.loss.type
        self.loss_function = LOSSES_LUT[loss_function_type](config)

        print("Preparing validation loaders")
        dataset_names = config.offline_validation.datasets
        self.offline_dataset_lut = {}
        for dataset_name in dataset_names:
            self.offline_dataset_lut[dataset_name] = DATALOADERS_LUT[dataset_name](
                config
            )

    @torch.no_grad()
    def shortlist(self, skip_if_done=True):
        if os.path.exists(self.shortlisted_data_path) and skip_if_done:
            print("Shortlist already computed, skipping")
            return

        indexes, confidences = self.shortlist_strategy.shortlist()

        assert os.path.exists(
            self.longlisted_data_path
        ), "All shortlist strategies must invoke subsample_dataset_for_train function"
        with open(self.longlisted_data_path) as f:
            longlist_rows = json.load(f)

        shortlisted_rows = []
        for idx, confidence in zip(indexes, confidences):
            row = longlist_rows[idx]
            row["confidence"] = confidence
            shortlisted_rows.append(row)

        with open(self.shortlisted_data_path, "w", encoding="utf-8") as f:
            json.dump(shortlisted_rows, f, indent=2)

    @torch.no_grad()
    def generate_few_shots(self, skip_if_done=True):
        if os.path.exists(self.few_shot_data_jsonl_path) and skip_if_done:
            print("few shots already computed, skipping")
            return

        try:
            with open(self.few_shot_data_jsonl_path, "w", encoding="utf-8") as f:
                for row, few_shot in self.shortlist_strategy.assemble_few_shot(
                    self.current_dataset_name
                ):
                    prompt = self.prompt_formatting_strategy.generate_prompt(
                        self.generative_model.tokenizer, row, few_shot
                    )
                    label = row["labels"]
                    line = json.dumps({"prompts": prompt, "labels": label})
                    f.write(line + "\n")
        except Exception as e:
            os.unlink(self.few_shot_data_jsonl_path)
            raise e

        self.generate_few_shot_data_sanity()

    def generate_few_shot_data_sanity(self):
        s = ""
        hr = "\n" + ("-" * 80) + "\n"
        with open(self.few_shot_data_jsonl_path, "r") as f:
            i = 0
            for row in f:
                row = json.loads(row)
                s += row["prompts"] + row["labels"] + hr
                i += 1
                if i == 5:
                    break

        with open(self.few_shot_data_sanity_path, "w", encoding="utf-8") as f:
            f.write(s)

    @torch.no_grad()
    def run_inference(self, skip_if_done=True):
        if os.path.exists(self.inference_result_jsonl_path) and skip_if_done:
            print("Inference already done")
            return

        wrapped_dataset = self.offline_dataset_lut[self.current_dataset_name]
        options = get_options_if_possible(wrapped_dataset)

        total = count_rows_jsonl(self.few_shot_data_jsonl_path)
        with open(self.few_shot_data_jsonl_path, "r") as f_in:
            with open(self.inference_result_jsonl_path, "w", encoding="utf-8") as f_out:
                for row in tqdm(f_in, total=total, desc="Running inference"):
                    row = json.loads(row)
                    prompt, true_answer = row["prompts"], row["labels"]

                    result = evaluate_with_options_if_possible(
                        self.generative_model, options, prompt, true_answer
                    )

                    f_out.write(json.dumps({**row, **result}))
                    f_out.write("\n")

    def analyze_inference_outputs(self):
        wrapped_dataset = self.offline_dataset_lut[self.current_dataset_name]
        is_mcq = hasattr(wrapped_dataset, "LABELS")
        if is_mcq:
            self.analyze_inference_outputs_mcq()
        else:
            self.analyze_inference_outputs_freeform()

    def analyze_inference_outputs_mcq(self):
        correct_count = 0
        total_probability = 0
        total = 0

        with open(self.inference_result_jsonl_path, "r") as file:
            for line in file:
                item = json.loads(line)  # Parse the JSON data from each line
                if item["correct"]:
                    correct_count += 1
                    label = item["labels"]
                    probability = item["option_probabilities"][label]
                    total_probability += probability
                total += 1

        accuracy = correct_count / total
        avg_correct_probability = total_probability / total

        # Write the results to an output file
        with open(self.final_result_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": accuracy,
                    "avg_correct_probability": avg_correct_probability,
                },
                f,
                indent=2,
            )

    def analyze_inference_outputs_freeform(self):
        total_count = 0
        correct_count = 0

        correct_predicted_sum = 0
        incorrect_predicted_sum = 0
        correct_predicted_count = 0
        incorrect_predicted_count = 0

        correct_target_sum = 0
        incorrect_target_sum = 0
        correct_target_count = 0
        incorrect_target_count = 0

        # Read the JSONL file line by line
        with open(self.inference_result_jsonl_path, "r") as file:
            for line in file:
                data = json.loads(line)  # Parse the JSON data from each line

                if data["predicted"] == data["target"]:
                    correct_count += 1
                    correct_predicted_sum += data["predicted_sequence_probability"]
                    correct_predicted_count += 1

                    correct_target_sum += data["target_sequence_probability"]
                    correct_target_count += 1
                else:
                    incorrect_predicted_sum += data["predicted_sequence_probability"]
                    incorrect_predicted_count += 1

                    incorrect_target_sum += data["target_sequence_probability"]
                    incorrect_target_count += 1

                total_count += 1

        def calculate_average(sum_value, count):
            return sum_value / count if count > 0 else 0

        correct_percentage = calculate_average(correct_count, total_count) * 100
        correct_predicted_avg = calculate_average(
            correct_predicted_sum, correct_predicted_count
        )
        incorrect_predicted_avg = calculate_average(
            incorrect_predicted_sum, incorrect_predicted_count
        )
        correct_target_avg = calculate_average(correct_target_sum, correct_target_count)
        incorrect_target_avg = calculate_average(
            incorrect_target_sum, incorrect_target_count
        )

        # Write the results to an output file
        with open(self.final_result_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "correct_percentage": correct_percentage,
                    "correct_predicted_avg": correct_predicted_avg,
                    "incorrect_predicted_avg": incorrect_predicted_avg,
                    "correct_target_avg": correct_target_avg,
                    "incorrect_target_avg": incorrect_target_avg,
                },
                f,
                indent=2,
            )

    @contextmanager
    def timer(self):
        try:
            start_time = time.time()  # Start timer
            yield  # Allow the block of code to execute
        finally:
            # End timer
            end_time = time.time()

            # Elapsed time in milliseconds
            elapsed_time = (end_time - start_time) * 1000

            # Log the time to a JSON file
            with open(Path(self.artifacts_dir) / "time_result.json", "w") as f:
                json.dump(
                    {
                        "start_timestamp": start_time,
                        "end_timestamp": end_time,
                        "elapsed_milliseconds": elapsed_time,
                    },
                    f,
                    indent=2,
                )

    def set_seed(self, seed):
        set_seed(seed)
        self.current_seed = seed

    @property
    def artifacts_dir(self):
        return generate_artifacts_dir(
            self.config, self.current_seed, self.current_dataset_name
        )

    @property
    def shortlisted_data_path(self):
        path = Path(self.artifacts_dir) / "shortlist.json"
        return path

    @property
    def longlisted_data_path(self):
        path = Path(self.artifacts_dir) / "longlist.json"
        return path

    @property
    def few_shot_data_jsonl_path(self):
        path = Path(self.artifacts_dir) / "few_shot_validation.jsonl"
        return path

    @property
    def few_shot_data_sanity_path(self):
        path = Path(self.artifacts_dir) / "few_shot_validation_sanity.txt"
        return path

    @property
    def inference_result_jsonl_path(self):
        path = Path(self.artifacts_dir) / "inference_result.jsonl"
        return path

    @property
    def final_result_json_path(self):
        path = Path(self.artifacts_dir) / "final_result.json"
        return path
