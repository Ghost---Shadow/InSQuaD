from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.training.batch_size
        self.dataset = None

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError()

    def get_loader(self, split):
        shuffle = False
        if split == "train":
            shuffle = True
        return DataLoader(
            self.dataset[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def random_access(self, indexes):
        SPLIT = "train"
        batch = []
        for index in indexes:
            batch.append(self.dataset[SPLIT][index])
        return self.collate_fn(batch)
