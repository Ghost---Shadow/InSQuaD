import json
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset


class DummyHotpotQaWithQDataset(HotpotQaWithQDataset):
    NAME = "dummy_hotpot_qa_with_q"

    def __init__(self, config):
        super().__init__(config)
        with open("src/dataloaders/paraphrase_sample.json") as f:
            row = json.load(f)
            self.dataset = {"train": [row], "validation": [row]}
