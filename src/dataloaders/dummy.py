from dataloaders.base import BaseDataset


class DummyDataset(BaseDataset):
    NAME = "dummy"

    def __init__(self, config):
        super().__init__(config)
        self.dataset = {
            "train": [
                {"prompt": "What is Alice's favourite fruit?", "label": "apple"},
                {"prompt": "What is Bob's favourite fruit?", "label": "banana"},
                {"prompt": "What is Charlie's favourite fruit?", "label": "coconut"},
            ],
            "validation": [
                {"prompt": "What is Mike's favourite fruit?", "label": "mango"},
            ],
        }

    @staticmethod
    def collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        labels = [item["label"] for item in batch]
        return {"prompts": prompts, "labels": labels}
