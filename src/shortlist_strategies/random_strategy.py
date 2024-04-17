from config import RootConfig
import numpy as np
from shortlist_strategies.base import BaseStrategy


class RandomStrategy(BaseStrategy):
    NAME = "random"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]
        indexes = np.arange(len(wrapped_dataset.dataset["train"]))
        np.random.shuffle(indexes)
        indexes = indexes[: self.config.offline_validation.annotation_budget]
        indexes = indexes.tolist()
        confidences = None  # No confidence metric for this strategy
        return indexes, confidences
