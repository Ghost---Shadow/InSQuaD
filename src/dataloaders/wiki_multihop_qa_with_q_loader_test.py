import unittest
from config import Config
from dataloaders.hotpot_qa_with_q_loader_test import row_test_inner
from dataloaders.wiki_multihop_qa_with_q_loader import WikiMultihopQaWithQDataset
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.wiki_multihop_qa_with_q_loader_test.TestWikiMultihopQaWithQLoader -v
class TestWikiMultihopQaWithQLoader(unittest.TestCase):
    # python -m unittest dataloaders.wiki_multihop_qa_with_q_loader_test.TestWikiMultihopQaWithQLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        config.training.batch_size = 1

        dataset = WikiMultihopQaWithQDataset(config)
        train_loader = dataset.get_loader(split="train")
        val_loader = dataset.get_loader(split="validation")

        # Train loader
        batch = next(iter(train_loader))
        question = "Are director of film All Square and director of film The Prize Fighter both from the same country?"
        sentences = [
            'Michael Preece( born September 15, 1936) is an American film and television director, script supervisor, producer, and actor best known for directing television series" Dallas" and" Walker, Texas Ranger" and films" The Prize Fighter" and.',
            "All Square is a 2018 American drama film directed by John Hyams.",
            "Directed by Michael Preece, it was written by Tim Conway and John Myhers, based on a story by Conway.",
            'John Hyams is an American screenwriter, director and cinematographer, best known for his involvement in the" Universal Soldier" series, for which he has directed two installments.',
        ]
        no_paraphrase_question = [
            "What are some of Michael Preece's notable works as a director, script supervisor, producer, and actor?",
            'Who directed the 2018 American drama film "All Square"?',
            "Who directed the film and who wrote the screenplay for it?",
            "What is John Hyams best known for in his career as a filmmaker?",
        ]
        paraphrased_questions = [
            "Which works by Michael Preece stand out in his career as a director, script supervisor, producer, and actor?",
            'Which filmmaker was in charge of directing the American drama film "All Square" released in 2018?',
            "Who was the director of the film and who was responsible for writing the screenplay?",
            "What is John Hyams most notable achievement as a filmmaker?",
        ]
        row_test_inner(
            batch,
            question,
            sentences,
            no_paraphrase_question,
            paraphrased_questions,
        )

        # Validation loader
        batch = next(iter(val_loader))
        question = (
            "Who is the mother of the director of film Polish-Russian War (Film)?"
        )
        sentences = [
            "(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.",
            "He is the son of actress Małgorzata Braunek and director Andrzej Żuławski.",
        ]
        no_paraphrase_question = [
            'What is the plot of the film "Wojna polsko-ruska" and who directed it?',
            "Who are the parents of the person mentioned in the passage?",
        ]
        paraphrased_questions = [
            'Who directed the film "Wojna polsko-ruska" and what is its storyline?',
            "Which individuals are identified as the parents of the individual referenced in the passage?",
        ]
        row_test_inner(
            batch, question, sentences, no_paraphrase_question, paraphrased_questions
        )

    # python -m unittest dataloaders.wiki_multihop_qa_with_q_loader_test.TestWikiMultihopQaWithQLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        config.training.batch_size = 2

        dataset = WikiMultihopQaWithQDataset(config)
        train_loader = dataset.get_loader(split="train")
        val_loader = dataset.get_loader(split="validation")
        for _ in tqdm(train_loader):
            ...

        for _ in tqdm(val_loader):
            ...


if __name__ == "__main__":
    unittest.main()
