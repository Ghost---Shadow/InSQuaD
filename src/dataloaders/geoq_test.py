import json
import unittest
from config import Config
from dataloaders.geoq import GeoQDataset
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.geoq_test.TestGeoQDatasetLoader -v
class TestGeoQDatasetLoader(unittest.TestCase):
    # python -m unittest dataloaders.geoq_test.TestGeoQDatasetLoader.test_geoq_loader -v
    def test_geoq_loader(self):
        # Set seed for deterministic testing
        set_seed(4)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        dataset = GeoQDataset(config)
        train_loader = dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))

        # with open("src/dataloaders/geoq_sample.json", "w") as f:
        #     json.dump(train_batch, f, indent=2)

        expected_batch = {
            "prompts": [
                "\nTable: border_info ['state_name', 'border']\nQuestion: how many states border the state that borders the most states\nQuery: "
            ],
            "labels": [
                "SELECT max(tmp.states) FROM(SELECT count(distinct border_info.border) AS states, border_info.state_name FROM border_info GROUP BY border_info.state_name) AS tmp;"
            ],
        }

        assert train_batch == expected_batch, train_batch

    # python -m unittest dataloaders.geoq_test.TestGeoQDatasetLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        dataset = GeoQDataset(config)

        train_loader = dataset.get_loader(split="train")
        for batch in tqdm(train_loader, desc="train"):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader, desc="validation"):
            assert "prompts" in batch
            assert "labels" in batch


if __name__ == "__main__":
    unittest.main()
