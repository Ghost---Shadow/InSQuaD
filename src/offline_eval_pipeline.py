import json
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
from train_utils import generate_artifacts_dir, set_seed


class OfflineEvaluationPipeline:
    def __init__(self, config: RootConfig):
        self.config = config
        self._load_parts(config)
        self.num_shots = config.offline_validation.num_shots
        self.current_seed = None

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

    def shortlist(self, dataset_name):
        indexes, confidences = self.shortlist_strategy.shortlist(dataset_name)

        dataset = self.offline_dataset_lut[dataset_name]
        shortlisted_rows = []
        for idx, confidence in zip(indexes, confidences):
            row = dataset[idx]
            row["confidence"] = confidence
            shortlisted_rows.append(row)

        with open(self.shortlisted_data_path, "w") as f:
            json.dump(shortlisted_rows, f, indent=2)

    def generate_one_shots(self, dataset_name):
        wrapped_dataset = self.offline_dataset_lut[dataset_name]
        with open(self.shortlisted_data_path) as f:
            shortlist = json.load(f)

        with open(self.few_shot_data_jsonl_path, "w") as f:
            for row, few_shot in self.shortlist_strategy.assemble_few_shot(
                wrapped_dataset, shortlist
            ):
                prompt = self.prompt_formatting_strategy(
                    self.generative_model.tokenizer, [row], [few_shot]
                )
                label = row["label"]
                line = json.dumps({"prompt": prompt, "label": label})
                f.write(line + "\n")

    def run_inference(self):
        counter = 0
        with open(self.few_shot_data_jsonl_path, "r") as f_in:
            while f_in:
                counter += 1
        with open(self.few_shot_data_jsonl_path, "r") as f_in:
            with open(self.inference_result_csv_path, "w") as f_out:
                f_out.write("expected,actual,ppl\n")
                for row in tqdm(f_in, total=counter, desc="Running inference"):
                    row = json.loads(row)
                    prompt, true_answer = row["prompt"], row["label"]
                    result = self.generative_model(prompt, true_answer)
                    sequence_probability, actual = (
                        result["sequence_probability"],
                        result["actual"],
                    )
                    f_out.write(f"{true_answer},{actual},{sequence_probability}\n")

    def set_seed(self, seed):
        set_seed(seed)
        self.current_seed = seed

    @property
    def artifacts_dir(self):
        return generate_artifacts_dir(self.config, self.current_seed)

    @property
    def shortlisted_data_path(self):
        path = Path(self.artifacts_dir) / "shortlist.json"
        return path

    @property
    def few_shot_data_jsonl_path(self):
        raise NotImplementedError()

    @property
    def inference_result_csv_path(self):
        raise NotImplementedError()
