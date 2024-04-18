import unittest
from config import Config
from dataloaders.mrpc import MRPC
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.mrpc_test.TestMRPCLoader -v
class TestMRPCLoader(unittest.TestCase):
    def test_mrpc_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        mrpc_dataset = MRPC(config)
        train_loader = mrpc_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        sentence1 = "Israel 's defense minister on Sunday raised the specter of an Israeli invasion in the Gaza Strip , where Palestinian militants already face a deadly air campaign ."
        sentence2 = "Shaul Mofaz , Israel 's Defence Minister , raised the prospect of a ground offensive into the Gaza Strip , in addition to a growing air campaign to assassinate Hamas militants ."
        expected_prompt = f"{sentence1}\n{sentence2}\n"
        expected_label = MRPC.LABELS[1]
        assert train_batch["prompts"][0] == expected_prompt
        assert train_batch["labels"][0] == expected_label

    # python -m unittest dataloaders.mrpc_test.TestMRPCLoader.test_random_access -v
    def test_random_access(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        mrpc_dataset = MRPC(config)

        indexes = [0, 1]
        batch = mrpc_dataset.random_access(indexes)

        expected_prompts = [
            'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .\nReferring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .\n',
            "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .\nYucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .\n",
        ]
        expected_labels = [
            "yes",
            "no",
        ]
        self.assertEqual(batch["prompts"], expected_prompts)
        self.assertEqual(batch["labels"], expected_labels)

    # python -m unittest dataloaders.mrpc_test.TestMRPCLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        dataset = MRPC(config)

        train_loader = dataset.get_loader(split="train")
        for batch in tqdm(train_loader):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader):
            assert "prompts" in batch
            assert "labels" in batch


if __name__ == "__main__":
    unittest.main()
