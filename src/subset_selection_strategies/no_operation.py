from subset_selection_strategies.base_strategy import BaseSubsetSelectionStrategy
import torch


class NoOperationSubsetSelectionStrategy(BaseSubsetSelectionStrategy):
    NAME = "noop"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline

    def subset_select(
        self, query_embedding: torch.Tensor, shortlist_embeddings: torch.Tensor
    ):
        size = shortlist_embeddings.shape[0]
        indices = torch.arange(size)
        scores = torch.zeros(size, dtype=torch.float32)

        return indices, scores
