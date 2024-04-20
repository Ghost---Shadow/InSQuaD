import json
from config import RootConfig
import numpy as np
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm


class RandomStrategy(BaseStrategy):
    NAME = "random"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, use_cache=True):
        longlist_rows = self.subsample_dataset_for_train()
        total = len(longlist_rows)

        indexes = np.arange(total)
        np.random.shuffle(indexes)
        indexes = indexes[: self.config.offline_validation.annotation_budget]
        indexes = indexes.tolist()
        confidences = [1 / total] * len(indexes)  # Uniform

        return indexes, confidences

    def assemble_few_shot(self, use_cache=True):
        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        eval_list_rows = self.subsample_dataset_for_eval()
        for row in tqdm(eval_list_rows, desc="Assembling few shot"):
            num_shots = self.config.offline_validation.num_shots
            few_shots = np.random.choice(shortlist, size=num_shots, replace=False)

            collated_few_shots = {"prompts": [], "labels": []}
            for few_shot in few_shots:
                collated_few_shots["prompts"].append(few_shot["prompts"])
                collated_few_shots["labels"].append(few_shot["labels"])

            yield row, collated_few_shots
