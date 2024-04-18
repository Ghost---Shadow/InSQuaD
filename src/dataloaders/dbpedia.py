from dataloaders.base import BaseDataset


class DBPedia(BaseDataset):
    LABELS = {
        0: "Company",
        1: "EducationalInstitution",
        2: "Artist",
        3: "Athlete",
        4: "OfficeHolder",
        5: "MeanOfTransportation",
        6: "Building",
        7: "NaturalPlace",
        8: "Village",
        9: "Animal",
        10: "Plant",
        11: "Album",
        12: "Film",
        13: "WrittenWork",
    }

    NAME = "dbpedia"

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(DBPedia.NAME, ("fancyzhx/dbpedia_14",))
        self.split_lut = {"train": "train", "validation": "test", "test": "test"}

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for item in batch:
            title, content, topic = (
                item["title"],
                item["content"],
                item["label"],
            )
            prompt = f"Title: {title}\nContent: {content}\nTopic:"
            prompts.append(prompt)
            labels.append(DBPedia.LABELS[topic])
        return {"prompts": prompts, "labels": labels}
