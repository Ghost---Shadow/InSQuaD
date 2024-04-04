from dataloaders.base import BaseDataset
import torch


class InMemory:
    def __init__(self, config):
        self.config = config

    def repopulate_index(self, wrapped_dataset: BaseDataset, embedding_model):
        raise NotImplementedError("TODO")

    def retrieve(self, query_embeddings: torch.Tensor):
        raise NotImplementedError("TODO")
