from dataloaders.base import BaseDataset
import torch


class InMemory:
    NAME = "in_memory"

    def __init__(self, config, pipeline):
        self.config = config
        self.config = pipeline

    def repopulate_index(self, wrapped_dataset: BaseDataset, embedding_model):
        raise NotImplementedError("TODO")

    def retrieve(self, query_embeddings: torch.Tensor, omit_self: bool):
        raise NotImplementedError("TODO")
