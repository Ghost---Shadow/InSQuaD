import unittest
from config import Config
from dataloaders.wiki_multihop_qa_loader import WikiMultihopQaDataset
from tqdm import tqdm
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.wiki_multihop_qa_loader_test.TestWikiMultihopQaLoader -v
class TestWikiMultihopQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.wiki_multihop_qa_loader_test.TestWikiMultihopQaLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        dataset = WikiMultihopQaDataset(config)
        train_loader = dataset.get_loader(split="train")

        # Train loader
        batch = next(iter(train_loader))
        expected = (
            "Are both Hanover Insurance and Broadway Video located in the same country?"
        )
        actual = batch["question"][0]
        assert actual == expected, actual

        expected = [
            "The Hanover Insurance Group, Inc., based in Worcester, Massachusetts, is one of the oldest continuous businesses in the United States still operating within its original industry.",
            'Broadway Video is an American multimedia entertainment studio founded by Lorne Michaels, creator of the sketch comedy TV series" Saturday Night Live" and producer of other television programs and movies.',
        ]
        actual = list(np.array(batch["documents"][0])[batch["relevant_indexes"][0]])
        assert expected == actual, actual
        assert sum(batch["correct_mask"][0]) == len(batch["relevant_indexes"][0])
        actual = list(np.array(batch["documents"][0])[batch["correct_mask"][0]])
        assert expected == actual, actual

        # Validation loader
        val_loader = dataset.get_loader(split="validation")
        batch = next(iter(val_loader))
        expected = (
            "Who is the mother of the director of film Polish-Russian War (Film)?"
        )
        actual = batch["question"][0]
        assert actual == expected, actual

        expected = [
            "(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.",
            "He is the son of actress Małgorzata Braunek and director Andrzej Żuławski.",
        ]
        actual = list(np.array(batch["documents"][0])[batch["relevant_indexes"][0]])
        assert expected == actual, actual

    # python -m unittest dataloaders.wiki_multihop_qa_loader_test.TestWikiMultihopQaLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 2
        dataset = WikiMultihopQaDataset(config)
        train_loader = dataset.get_loader(split="train")
        val_loader = dataset.get_loader(split="validation")

        for _ in tqdm(train_loader):
            ...

        for _ in tqdm(val_loader):
            ...


if __name__ == "__main__":
    unittest.main()
