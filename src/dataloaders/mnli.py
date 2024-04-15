from dataloaders.base import BaseDataset


class MNLI(BaseDataset):
    # neutral, entailment, contradiction
    LABELS = {0: "maybe", 1: "yes", 2: "no"}
    NAME = "mnli"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(MNLI.NAME, ("glue", "mnli"))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            premise, hypothesis, label = (
                item["premise"],
                item["hypothesis"],
                item["label"],
            )
            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
            prompts.append(prompt)
            labels.append(MNLI.LABELS[label])
        return {"prompts": prompts, "labels": labels}

    def get_loader(self, split):
        if split == "validation":
            split = "validation_matched"
        return super().get_loader(split)
