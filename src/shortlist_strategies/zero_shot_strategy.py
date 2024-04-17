from config import RootConfig
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm


class ZeroShotStrategy(BaseStrategy):
    NAME = "zero_shot"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, dataset_name, use_cache=True):
        indexes = []
        confidences = []
        return indexes, confidences

    def assemble_few_shot(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]
        total, subsampled_validation_iterator = self.subsample_dataset(
            wrapped_dataset, "validation"
        )
        for row in tqdm(
            subsampled_validation_iterator,
            desc="Assembling few shot",
            total=total,
        ):
            collated_few_shots = {"prompts": [], "labels": []}
            yield row, collated_few_shots
