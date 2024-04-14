from collections import defaultdict
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
        keys = list(batch[0].keys())
        collated_batch = defaultdict(list)

        for key in keys:
            for row in batch:
                collated_batch[key].append(row[key])

        return dict(collated_batch)

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

    def get_row_iterator(self, split):
        # TODO: Cleanup
        for raw_row in self.dataset[split]:
            batch = self.collate_fn([raw_row])
            row = self.unbatch(batch)
            yield row

    def unbatch(self, batch):
        # TODO: Cleanup
        row = {}
        for key in batch:
            assert len(batch[key]) == 1
            row[key] = batch[key][0]
        return row

    def random_access(self, indexes):
        SPLIT = "train"
        batch = []
        for index in indexes:
            batch.append(self.dataset[SPLIT][index])
        return self.collate_fn(batch)

    def __getitem__(self, index):
        batch = self.random_access([index])
        return self.unbatch(batch)
