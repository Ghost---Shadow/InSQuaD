import unittest
from config import Config
from dataloaders.dataloaders_test_utils import test_if_one_token
from dataloaders.rte import RTE
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.rte_test.TestRTELoader -v
class TestRTELoader(unittest.TestCase):
    # python -m unittest dataloaders.rte_test.TestRTELoader.test_rte_loader -v
    def test_rte_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        rte_dataset = RTE(config)
        train_loader = rte_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        expected_prompt = "Premise: Three Democrats joined the committee's 10 majority Republicans in a 13-5 vote to advance the conservative judge's nomination to the full Senate. Five Democrats opposed Roberts.\nHypothesis: The Senate Judiciary Committee, on Thursday, approved Judge John Roberts' nomination as the next Supreme Court Chief Justice, virtually assuring his confirmation by the Senate next week.\nAnswer:"
        expected_label = RTE.LABELS[1]
        assert train_batch["prompts"][0] == expected_prompt, (
            "(" + train_batch["prompts"][0] + ")"
        )
        assert train_batch["labels"][0] == expected_label, train_batch["labels"][0]

    # python -m unittest dataloaders.rte_test.TestRTELoader.test_no_crash -v
    def test_no_crash(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        rte_dataset = RTE(config)

        train_loader = rte_dataset.get_loader(split="train")
        for batch in tqdm(train_loader):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = rte_dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader):
            assert "prompts" in batch
            assert "labels" in batch

    # python -m unittest dataloaders.rte_test.TestRTELoader.test_labels_one_token -v
    def test_labels_one_token(self):
        test_if_one_token(RTE)


if __name__ == "__main__":
    unittest.main()
