from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk


class BaseDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.training.batch_size
        self.dataset = None

    def cached_load_dataset(self, name, source):
        """
        https://github.com/huggingface/datasets/issues/824
        """
        cache_path = f"./artifacts/data_cache/{name}"
        Path(cache_path).mkdir(exist_ok=True, parents=True)
        try:
            self.dataset = load_from_disk(cache_path)
        except FileNotFoundError:
            self.dataset = load_dataset(*source)
            self.dataset.save_to_disk(cache_path)

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

    def __getitem__(self, index):
        batch = self.random_access([index])
        row = {}
        for key in batch:
            row[key] = batch[key][0]

        return row
