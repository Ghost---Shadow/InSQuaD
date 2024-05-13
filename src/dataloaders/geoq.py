import json
from dataloaders.base import BaseDataset
from prompt_formatting_strategies.bare import BareStrategy


class GeoQDataset(BaseDataset):
    LABELS = None

    NAME = "geoq"
    OVERRIDE_PROMPT_FORMATTING_STRATEGY = BareStrategy.NAME

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(GeoQDataset.NAME, ("vaishali/geoQuery-tableQA",))

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []

        for row in batch:
            question = row["question"]
            question = f"Question: {question}\n"

            table_info = ""
            for table_name, table in zip(row["table_names"], row["tables"]):
                table = json.loads(table)
                columns = table["columns"]
                table_info += f"Table: {table_name} {columns}\n"

            prompt = f"{table_info}{question}Query: "
            label = row["query"]

            assert label.startswith("SELECT") or label.startswith("(SELECT"), label

            prompts.append("\n" + prompt)
            labels.append(label)

        return {"prompts": prompts, "labels": labels}
