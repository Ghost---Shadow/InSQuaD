from subset_selection_strategies.base_strategy import BaseSubsetSelectionStrategy
import torch


class TopKStrategy(BaseSubsetSelectionStrategy):
    def __init__(self, config):
        self.config = config
        self.k = config.architecture.subset_selection_strategy.k

    def get_indexes_from_qd(self, quality_vector, diversity_matrix):
        # Should be already sorted
        # Tile the arange sequence with the batch size
        return torch.tile(torch.arange(self.k), (quality_vector.size(0), 1))
