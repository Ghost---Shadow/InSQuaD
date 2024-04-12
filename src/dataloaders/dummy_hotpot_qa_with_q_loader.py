import json
from dataloaders.base import BaseDataset
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset


class DummyHotpotQaWithQDataset(HotpotQaWithQDataset):
    NAME = "dummy_hotpot_qa_with_q"

    def __init__(self, config):
        BaseDataset.__init__(self, config)
        with open("src/dataloaders/paraphrase_sample.json") as f:
            row = json.load(f)
            self.dataset = {"train": [row], "validation": [row]}
