from dataloaders.base import BaseDataset


class RTE(BaseDataset):
    # entailment, not entailment
    LABELS = {0: "yes", 1: "no"}
    NAME = "rte"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(RTE.NAME, ("glue", "rte"))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            premise, hypothesis, label = (
                item["sentence1"],
                item["sentence2"],
                item["label"],
            )
            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
            prompts.append(prompt)
            labels.append(RTE.LABELS[label])
        return {"prompts": prompts, "labels": labels}
