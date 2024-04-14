import json
import os
from pathlib import Path
from config import RootConfig
from dataloaders import DATALOADERS_LUT
from dense_indexes import DENSE_INDEXES_LUT
from generative_models import GENERATIVE_MODELS_LUT
from losses import LOSSES_LUT
from prompt_formatting_strategies import PROMPT_FORMATTING_STRATEGIES_LUT
from semantic_search_models import SEMANTIC_SEARCH_MODELS_LUT
from shortlist_strategies import SHORTLIST_STRATEGIES_LUT
from subset_selection_strategies import SUBSET_SELECTION_STRATEGIES_LUT
from tqdm import tqdm
from train_utils import count_rows_jsonl, generate_artifacts_dir, set_seed


class OfflineEvaluationPipeline:
    def __init__(self, config: RootConfig):
        self.config = config
        self._load_parts(config)
        self.num_shots = config.offline_validation.num_shots
        self.current_seed = None
        self.current_dataset_name = None

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
        # TODO: Immediately load latest checkpoint

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

    def shortlist(self, skip_if_done=True):
        dataset_name = self.current_dataset_name
        if os.path.exists(self.shortlisted_data_path) and skip_if_done:
            print("Shortlist already computed, skipping")
            return

        indexes, confidences = self.shortlist_strategy.shortlist(dataset_name)

        dataset = self.offline_dataset_lut[dataset_name]
        shortlisted_rows = []
        for idx, confidence in zip(indexes, confidences):
            row = dataset[idx]
            row["confidence"] = confidence
            shortlisted_rows.append(row)

        with open(self.shortlisted_data_path, "w") as f:
            json.dump(shortlisted_rows, f, indent=2)

    def generate_few_shots(self, skip_if_done=True):
        if os.path.exists(self.few_shot_data_jsonl_path) and skip_if_done:
            print("few shots already computed, skipping")
            return

        try:
            with open(self.few_shot_data_jsonl_path, "w") as f:
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

        with open(self.few_shot_data_sanity_path, "w") as f:
            f.write(s)

    def run_inference(self, skip_if_done=True):
        if os.path.exists(self.final_result_json_path) and skip_if_done:
            print("Inference already done")
            return

        total = count_rows_jsonl(self.few_shot_data_jsonl_path)
        with open(self.few_shot_data_jsonl_path, "r") as f_in:
            with open(self.inference_result_jsonl_path, "w") as f_out:
                for row in tqdm(f_in, total=total, desc="Running inference"):
                    row = json.loads(row)
                    prompt, true_answer = row["prompts"], row["labels"]
                    result = self.generative_model.evaluate(prompt, true_answer)
                    f_out.write(json.dumps({**row, **result}))
                    f_out.write("\n")

        self.analyze_inference_outputs()

    def analyze_inference_outputs(self):
        total_count = 0
        correct_count = 0
        correct_prob_sum = 0
        incorrect_prob_sum = 0
        correct_prob_count = 0
        incorrect_prob_count = 0

        # Read the JSONL file line by line
        with open(self.inference_result_jsonl_path, "r") as file:
            for line in file:
                data = json.loads(line)  # Parse the JSON data from each line

                if data["labels"] == data["actual"]:
                    correct_count += 1
                    correct_prob_sum += data["sequence_probability"]
                    correct_prob_count += 1
                else:
                    incorrect_prob_sum += data["sequence_probability"]
                    incorrect_prob_count += 1

                total_count += 1

        # Calculate the percentages and averages
        if total_count > 0:
            correct_percentage = (correct_count / total_count) * 100
        else:
            correct_percentage = 0

        if correct_prob_count > 0:
            average_correct_prob = correct_prob_sum / correct_prob_count
        else:
            average_correct_prob = 0

        if incorrect_prob_count > 0:
            average_incorrect_prob = incorrect_prob_sum / incorrect_prob_count
        else:
            average_incorrect_prob = 0

        # Write the results to an output file
        with open(self.final_result_json_path, "w") as f:
            json.dump(
                {
                    "correct_percentage": correct_percentage,
                    "average_correct_prob": average_correct_prob,
                    "average_incorrect_prob": average_incorrect_prob,
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
