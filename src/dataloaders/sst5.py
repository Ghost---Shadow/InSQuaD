from dataloaders.base import BaseDataset


class SST5(BaseDataset):
    LABELS = {
        0: "very negative",
        1: "negative",
        2: "neutral",
        3: "positive",
        4: "very positive",
    }
    NAME = "sst5"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(SST5.NAME, ("SetFit/sst5",))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            sentence, label = (
                item["text"],
                item["label"],
            )
            prompt = sentence
            prompts.append(prompt)
            labels.append(SST5.LABELS[label])
        return {"prompts": prompts, "labels": labels}
