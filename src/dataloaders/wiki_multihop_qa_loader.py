from dataloaders.base import BaseDataset
from dataloaders.hotpot_qa_loader import HotpotQaDataset
from datasets import load_dataset


class WikiMultihopQaDataset(BaseDataset):
    NAME = "wiki_multihop_qa"

    def __init__(self, config):
        super().__init__(config)
        self.dataset = load_dataset("scholarly-shadows-syndicate/2WikiMultihopQA")

    @staticmethod
    def collate_fn(batch):
        return HotpotQaDataset.collate_fn(batch, KEY_SENTENCES="content")

    def get_loader(self, split):
        if split == "validation":
            split = "dev"
        return super().get_loader(split)
