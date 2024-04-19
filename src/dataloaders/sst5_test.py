import unittest
from config import Config
from dataloaders.dataloaders_test_utils import test_if_one_token
from dataloaders.sst5 import SST5
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.sst5_test.TestSST5Loader -v
class TestSST5Loader(unittest.TestCase):
    # python -m unittest dataloaders.sst5_test.TestSST5Loader.test_sst5_loader -v
    def test_sst5_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        sst5_dataset = SST5(config)
        train_loader = sst5_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        sentence = (
            "my little eye is the best little `` horror '' movie i 've seen in years ."
        )
        expected_prompt = sentence
        expected_label = SST5.LABELS[4]
        assert train_batch["prompts"][0] == expected_prompt, (
            "(" + train_batch["prompts"][0] + ")"
        )
        assert train_batch["labels"][0] == expected_label, train_batch["labels"][0]

    # python -m unittest dataloaders.sst5_test.TestSST5Loader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        sst5_dataset = SST5(config)

        train_loader = sst5_dataset.get_loader(split="train")
        for batch in tqdm(train_loader):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = sst5_dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader):
            assert "prompts" in batch
            assert "labels" in batch

    # python -m unittest dataloaders.sst5_test.TestSST5Loader.test_no_bad_labels -v
    def test_no_bad_labels(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        sst5_dataset = SST5(config)

        for split in ["train", "validation"]:
            for row in tqdm(sst5_dataset.dataset[split]):
                assert row["label_text"] == SST5.LABELS[row["label"]], row["label_text"]


if __name__ == "__main__":
    unittest.main()
