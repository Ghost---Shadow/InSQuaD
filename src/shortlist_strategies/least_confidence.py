import json
from config import RootConfig
from eval_utils import evaluate_with_options_if_possible, get_options_if_possible
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm


class LeastConfidenceStrategy(BaseStrategy):
    NAME = "least_confidence"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, use_cache=True):
        dataset_name = self.pipeline.current_dataset_name
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]
        longlist_rows = self.subsample_dataset_for_train()

        options = get_options_if_possible(wrapped_dataset)

        all_confidences = []
        for row in tqdm(longlist_rows, desc="Computing confidences"):
            prompt, true_answer = row["prompts"], row["labels"]

            result = evaluate_with_options_if_possible(
                self.pipeline.generative_model, options, prompt, true_answer
            )

            if options is None:
                confidence = result["target_sequence_probability"]
            else:
                confidence = result["option_probabilities"][true_answer]

            all_confidences.append(confidence)

        indexes = sorted(
            range(len(all_confidences)), key=lambda i: all_confidences[i], reverse=False
        )
        indexes = indexes[: self.config.offline_validation.annotation_budget]

        confidences = [all_confidences[i] for i in indexes]

        return indexes, confidences

    def assemble_few_shot(self, use_cache=True):
        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        eval_list_rows = self.subsample_dataset_for_eval()

        # Pick globally top-n least confident for all rows
        num_shots = self.config.offline_validation.num_shots
        few_shots = shortlist[:num_shots]

        for row in tqdm(eval_list_rows, desc="Assembling few shot"):
            collated_few_shots = {"prompts": [], "labels": []}
            for few_shot in few_shots:
                collated_few_shots["prompts"].append(few_shot["prompts"])
                collated_few_shots["labels"].append(few_shot["labels"])

            yield row, collated_few_shots
