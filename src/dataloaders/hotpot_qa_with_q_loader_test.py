import unittest
from config import Config
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset
from tqdm import tqdm
from train_utils import set_seed
import numpy as np


def row_test_inner(
    batch, question, sentences, no_paraphrase_question, paraphrased_questions
):
    expected = question
    actual = batch["question"][0]
    assert actual == expected, actual

    expected = sentences
    actual = list(
        np.array(batch["upstream_documents"][0])[batch["relevant_indexes"][0]]
    )
    assert expected == actual, actual

    no_paraphrase_expected = no_paraphrase_question

    expected = [
        *no_paraphrase_expected,
        *paraphrased_questions,
    ]
    actual = list(np.array(batch["documents"][0])[batch["relevant_indexes"][0]])
    assert no_paraphrase_expected == actual, actual
    actual = list(np.array(batch["documents"][0])[batch["correct_mask"][0]])
    assert expected == actual, actual

    assert (
        len(batch["relevant_indexes"][0]) * 2
        == np.array(batch["correct_mask"][0]).sum()
    )

    paraphrase_masks = batch["paraphrase_masks"][0]
    documents = np.array(batch["documents"][0])

    for (left_mask, right_mask), original, paraphrased in zip(
        paraphrase_masks, no_paraphrase_expected, paraphrased_questions
    ):
        assert original == documents[left_mask], documents[left_mask]
        assert paraphrased == documents[right_mask], documents[right_mask]


# python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader -v
class TestHotpotQaWithQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        config.training.batch_size = 1

        dataset = HotpotQaWithQDataset(config)
        train_loader = dataset.get_loader(split="train")
        val_loader = dataset.get_loader(split="validation")

        # Train loader
        batch = next(iter(train_loader))
        question = "Who was born first, William March or Richard Brautigan?"
        sentences = [
            "William March (September 18, 1893 – May 15, 1954) was an American writer of psychological fiction and a highly decorated US Marine.",
            "Richard Gary Brautigan (January 30, 1935 – ca.",
        ]
        no_paraphrase_question = [
            "What were some notable achievements of William March as both a writer and a US Marine?",
            "What is the significance of Richard Brautigan's work in the literary world?",
        ]
        paraphrased_questions = [
            "What were William March's accomplishments and contributions as an American writer and US Marine?",
            "Can you rephrase the question about Richard Gary Brautigan?",
        ]
        row_test_inner(
            batch, question, sentences, no_paraphrase_question, paraphrased_questions
        )

        # Validation loader
        batch = next(iter(val_loader))
        question = "Were Scott Derrickson and Ed Wood of the same nationality?"
        sentences = [
            "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
            "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.",
        ]
        no_paraphrase_question = [
            "What is Scott Derrickson known for in the entertainment industry?",
            "What were some of the notable contributions and achievements of Edward Davis Wood Jr. in the field of filmmaking?",
        ]
        paraphrased_questions = [
            "What is Scott Derrickson known for?",
            "Who was Edward Davis Wood Jr.?",
        ]
        row_test_inner(
            batch, question, sentences, no_paraphrase_question, paraphrased_questions
        )

    # python -m unittest dataloaders.hotpot_qa_loader_test.TestHotpotQaLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # https://github.com/hotpotqa/hotpot/issues/47
        config = Config.from_file("experiments/dummy_experiment.yaml")
        config.training.batch_size = 2

        dataset = HotpotQaWithQDataset(config)
        train_loader = dataset.get_loader(split="train")
        val_loader = dataset.get_loader(split="validation")
        for _ in tqdm(train_loader):
            ...

        for _ in tqdm(val_loader):
            ...


if __name__ == "__main__":
    unittest.main()
