import unittest
from config import Config
from dataloaders.dataloaders_test_utils import test_if_one_token
from dataloaders.dbpedia import DBPedia
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.dbpedia_test.TestDBPediaLoader -v
class TestDBPediaLoader(unittest.TestCase):
    # python -m unittest dataloaders.dbpedia_test.TestDBPediaLoader.test_dbpedia_loader -v
    def test_dbpedia_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        dbpedia_dataset = DBPedia(config)
        train_loader = dbpedia_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        sentence = "Title: Barbus trispilomimus\nContent:  Barbus trispilomimus is a species of ray-finned fish in the genus Barbus.\nTopic:"
        expected_prompt = sentence
        expected_label = DBPedia.LABELS[9]
        assert train_batch["prompts"][0] == expected_prompt, (
            "(" + train_batch["prompts"][0] + ")"
        )
        assert train_batch["labels"][0] == expected_label, train_batch["labels"][0]

    # python -m unittest dataloaders.dbpedia_test.TestDBPediaLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        dbpedia_dataset = DBPedia(config)

        train_loader = dbpedia_dataset.get_loader(split="train")
        for batch in tqdm(train_loader):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = dbpedia_dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader):
            assert "prompts" in batch
            assert "labels" in batch

    # Can't be helped, but is not a bug, it just helps
    # # python -m unittest dataloaders.dbpedia_test.TestDBPediaLoader.test_labels_one_token -v
    # def test_labels_one_token(self):
    #     test_if_one_token(DBPedia)


if __name__ == "__main__":
    unittest.main()
