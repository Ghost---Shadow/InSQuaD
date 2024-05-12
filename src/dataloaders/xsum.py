from dataloaders.base import BaseDataset


class XsumDataset(BaseDataset):
    LABELS = None

    NAME = "xsum"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(XsumDataset.NAME, ("EdinburghNLP/xsum",))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            document, summary = (
                item["document"],
                item["summary"],
            )
            document = document.replace("\n", "")
            summary = summary.replace("\n", "")
            prompt = f"Document: {document}\nSummary:"
            prompts.append(prompt)
            labels.append(summary)
        return {"prompts": prompts, "labels": labels}
