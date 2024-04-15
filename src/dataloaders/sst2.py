from dataloaders.base import BaseDataset


class SST2(BaseDataset):
    LABELS = {0: "negative", 1: "positive"}
    NAME = "sst2"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(SST2.NAME, ("glue", "sst2"))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            sentence, label = (
                item["sentence"],
                item["label"],
            )
            prompt = sentence
            prompts.append(prompt)
            labels.append(SST2.LABELS[label])
        return {"prompts": prompts, "labels": labels}
