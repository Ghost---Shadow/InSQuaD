import unittest
from config import Config
from dataloaders.dataloaders_test_utils import test_if_one_token
from dataloaders.hellaswag import Hellaswag
from tqdm import tqdm
from train_utils import set_seed


expected_prompt = """
Context: The song slows and builds as the crowd joins back in with claps. The song ends and the crowd applauds thunderously. the saxophonist
a) stands on one foot in a pose and does steps on the microphone.
b) bows and returns to her seat.
c) sets the guitar aside and gestures at the camera.
d) bows over the microphone and salutes his group in the crowd.
Answer:
""".strip()


# python -m unittest dataloaders.hellaswag_test.TestHellaswagLoader -v
class TestHellaswagLoader(unittest.TestCase):
    # python -m unittest dataloaders.hellaswag_test.TestHellaswagLoader.test_hellaswag_loader -v
    def test_hellaswag_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        hellaswag_dataset = Hellaswag(config)
        train_loader = hellaswag_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        expected_label = Hellaswag.LABELS[1]
        assert train_batch["prompts"][0] == expected_prompt, (
            "(" + train_batch["prompts"][0] + ")"
        )
        assert train_batch["labels"][0] == expected_label, train_batch["labels"][0]

    # python -m unittest dataloaders.hellaswag_test.TestHellaswagLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        hellaswag_dataset = Hellaswag(config)

        train_loader = hellaswag_dataset.get_loader(split="train")
        for batch in tqdm(train_loader):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = hellaswag_dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader):
            assert "prompts" in batch
            assert "labels" in batch

    # python -m unittest dataloaders.hellaswag_test.TestHellaswagLoader.test_labels_one_token -v
    def test_labels_one_token(self):
        test_if_one_token(Hellaswag)


if __name__ == "__main__":
    unittest.main()
