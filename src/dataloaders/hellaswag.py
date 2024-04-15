from dataloaders.base import BaseDataset


class Hellaswag(BaseDataset):
    LABELS = {
        0: "a",
        1: "b",
        2: "c",
        3: "d",
    }

    NAME = "hellaswag"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(Hellaswag.NAME, ("Rowan/hellaswag",))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            context, continuations, label = (
                item["ctx"],
                item["endings"],
                item["label"],
            )
            label = int(label)
            options = ["a", "b", "c", "d"]
            assert len(options) >= len(continuations), len(continuations)
            choices = ""
            for option, continuation in zip(options, continuations):
                choices += f"{option}) {continuation}\n"
            prompt = f"Context: {context}\n{choices}Answer:"
            prompts.append(prompt)
            labels.append(Hellaswag.LABELS[label])
        return {"prompts": prompts, "labels": labels}
