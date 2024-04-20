from config import RootConfig
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm


class ZeroShotStrategy(BaseStrategy):
    NAME = "zero_shot"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, use_cache=True):
        # This creates long_list.json
        self.subsample_dataset_for_train()

        indexes = []
        confidences = []
        return indexes, confidences

    def assemble_few_shot(self, use_cache=True):
        eval_list_rows = self.subsample_dataset_for_eval()
        for row in tqdm(eval_list_rows, desc="Assembling few shot"):
            collated_few_shots = {"prompts": [], "labels": []}
            yield row, collated_few_shots
