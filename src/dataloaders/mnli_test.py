import unittest
from config import Config
from dataloaders.dataloaders_test_utils import test_if_one_token
from dataloaders.mnli import MNLI
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.mnli_test.TestMNLILoader -v
class TestMNLILoader(unittest.TestCase):
    # python -m unittest dataloaders.mnli_test.TestMNLILoader.test_mnli_loader -v
    def test_mnli_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        mnli_dataset = MNLI(config)
        train_loader = mnli_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        expected_prompt = "Premise: She said that the prevalence of alcohol problems was higher than other risk factors.\nHypothesis: She said alcohol problems were less likely.\nAnswer:"
        expected_label = MNLI.LABELS[2]
        assert train_batch["prompts"][0] == expected_prompt, (
            "(" + train_batch["prompts"][0] + ")"
        )
        assert train_batch["labels"][0] == expected_label, train_batch["labels"][0]

    # python -m unittest dataloaders.mnli_test.TestMNLILoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        mnli_dataset = MNLI(config)

        train_loader = mnli_dataset.get_loader(split="train")
        for batch in tqdm(train_loader):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = mnli_dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader):
            assert "prompts" in batch
            assert "labels" in batch

    # python -m unittest dataloaders.mnli_test.TestMNLILoader.test_labels_one_token -v
    def test_labels_one_token(self):
        test_if_one_token(MNLI)


if __name__ == "__main__":
    unittest.main()
