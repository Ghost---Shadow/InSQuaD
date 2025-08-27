import torch


class NoOperation:
    NAME = "noop"

    def __init__(self, config, pipeline):
        self.pipeline = pipeline
        self.config = config

    def before_each_epoch(self):
        ...

    def train_step(self, batch):
        return torch.tensor(0, requires_grad=True)
