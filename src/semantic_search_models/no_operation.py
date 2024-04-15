from semantic_search_models.base import WrappedBaseModel
import torch


class DummyModel:
    def parameters(self):
        return [torch.tensor([0.0], requires_grad=True)]


class NoOp(WrappedBaseModel):
    NAME = "noop"

    def __init__(self, config):
        super().__init__(config)
        self.model = DummyModel()
