import unittest
from config import Config
from dataloaders.dataloaders_test_utils import test_if_one_token
from dataloaders.sst2 import SST2
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.sst2_test.TestSST2Loader -v
class TestSST2Loader(unittest.TestCase):
    # python -m unittest dataloaders.sst2_test.TestSST2Loader.test_sst2_loader -v
    def test_sst2_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        sst2_dataset = SST2(config)
        train_loader = sst2_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        sentence = "more revealing , more emotional "
        expected_prompt = sentence
        expected_label = SST2.LABELS[1]
        assert train_batch["prompts"][0] == expected_prompt, (
            "(" + train_batch["prompts"][0] + ")"
        )
        assert train_batch["labels"][0] == expected_label, train_batch["labels"][0]

    # python -m unittest dataloaders.sst2_test.TestSST2Loader.test_get_item -v
    def test_get_item(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        sst2_dataset = SST2(config)

        for row in tqdm(sst2_dataset):
            assert "prompts" in row
            assert "labels" in row

    # python -m unittest dataloaders.sst2_test.TestSST2Loader.test_labels_one_token -v
    def test_labels_one_token(self):
        test_if_one_token(SST2)


if __name__ == "__main__":
    unittest.main()
