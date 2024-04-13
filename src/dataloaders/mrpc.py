from dataloaders.base import BaseDataset


class MRPC(BaseDataset):
    LABELS = {0: "not equivalent", 1: "equivalent"}
    NAME = "mrpc"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(MRPC.NAME, ("glue", "mrpc"))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            sentence1, sentence2, label = (
                item["sentence1"],
                item["sentence2"],
                item["label"],
            )
            prompt = f"{sentence1}\n{sentence2}\n"
            prompts.append(prompt)
            labels.append(MRPC.LABELS[label])
        return {"prompts": prompts, "labels": labels}
