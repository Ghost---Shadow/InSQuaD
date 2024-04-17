import json
from config import RootConfig
import numpy as np
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm


class RandomStrategy(BaseStrategy):
    NAME = "random"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]
        total_dataset_length = len(wrapped_dataset.dataset["train"])
        indexes = np.arange(total_dataset_length)
        np.random.shuffle(indexes)
        indexes = indexes[: self.config.offline_validation.annotation_budget]
        indexes = indexes.tolist()
        confidences = [1 / total_dataset_length] * len(indexes)  # Uniform
        return indexes, confidences

    def assemble_few_shot(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]

        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        total, subsampled_validation_iterator = self.subsample_dataset(
            wrapped_dataset, "validation"
        )
        for row in tqdm(
            subsampled_validation_iterator,
            desc="Assembling few shot",
            total=total,
        ):
            num_shots = self.config.offline_validation.num_shots
            few_shots = np.random.choice(shortlist, size=num_shots, replace=False)

            collated_few_shots = {"prompts": [], "labels": []}
            for few_shot in few_shots:
                collated_few_shots["prompts"].append(few_shot["prompts"])
                collated_few_shots["labels"].append(few_shot["labels"])

            yield row, collated_few_shots
