from dataloaders.base import BaseDataset


class InMemoryDataset(BaseDataset):
    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = {"train": dataset}
