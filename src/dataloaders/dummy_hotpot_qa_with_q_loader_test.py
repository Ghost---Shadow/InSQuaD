import unittest
from config import Config
from dataloaders.dummy_hotpot_qa_with_q_loader import DummyHotpotQaWithQDataset
from dataloaders.hotpot_qa_with_q_loader_test import row_test_inner


# python -m unittest dataloaders.dummy_hotpot_qa_with_q_loader_test.TestDummyHotpotQaWithQDataset -v
class TestDummyHotpotQaWithQDataset(unittest.TestCase):
    # python -m unittest dataloaders.dummy_hotpot_qa_with_q_loader_test.TestDummyHotpotQaWithQDataset.test_happy_path -v
    def test_happy_path(self):
        config = Config.from_file("experiments/dummy_experiment.yaml")
        config.training.batch_size = 1

        dataset = DummyHotpotQaWithQDataset(config)
        train_loader = dataset.get_loader(split="train")
        val_loader = dataset.get_loader(split="validation")

        # Train loader
        batch = next(iter(train_loader))
        question = 'The man Tony Bennet called "The Father of Rock and Roll" once toured with what singer who once auditioned for "Junior Junction", and landed a recording contract in her teens?'
        sentences = [
            "John Alvin Ray (January 10, 1927 – February 24, 1990) was an American singer, songwriter, and pianist.",
            " Extremely popular for most of the 1950s, Ray has been cited by critics as a major precursor to what would become rock and roll, for his jazz and blues-influenced music and his animated stage personality.",
            ' Tony Bennett called Ray the "father of rock and roll," and historians have noted him as a pioneering figure in the development of the genre.',
            "Lola Dee is an American singer and recording artist with Mercury Records and Columbia Records labels in the 1950s and 1960s.",
            ' At the age of 14, she was heard in an amateur contest and asked to audition for a network teen-aged show called "Junior Junction".',
            " At 16 she was signed to a recording contract.",
            ' She recorded over 40 sides, including the half million best seller "Only You" in 1955.',
            " Her popularity as a recording artist gave her the opportunity to tour with such stars as Bob Hope, Johnnie Ray and Jimmy Durante in the late 1950s and 1960s.",
        ]
        no_paraphrase_question = [
            "What were John Alvin Ray's main contributions to the music industry?",
            "How did Ray's jazz and blues-influenced music and animated stage personality contribute to the development of rock and roll?",
            'Who is considered the "father of rock and roll" and why?',
            "What record labels did Lola Dee work with during her career as a singer and recording artist?",
            "What opportunity did the girl receive after being heard in an amateur contest at the age of 14?",
            "What age was she when she signed a recording contract?",
            "What was the title of the half million best-selling record that she recorded in 1955?",
            "Who were some of the stars that she toured with in the late 1950s and 1960s?",
        ]
        paraphrased_questions = [
            "Who was John Alvin Ray?",
            "Ray was highly acclaimed and widely loved during the majority of the 1950s. Critics have recognized him as a significant influence on the development of rock and roll due to his music, which was influenced by jazz and blues, and his lively stage presence.",
            "What is Ray's role in the development of rock and roll, as recognized by Tony Bennett and historians?",
            "What record labels did Lola Dee work with in the 1950s and 1960s as an American singer and recording artist?",
            'When she was 14 years old, she participated in an amateur contest and was invited to audition for a television show called "Junior Junction" that was specifically for teenagers.',
            "When she was 16 years old, she secured a recording contract.",
            'In 1955, she achieved great success with her record "Only You," which sold over half a million copies, and she went on to record more than 40 songs.',
            "In the late 1950s and 1960s, she had the chance to go on tour with famous celebrities like Bob Hope, Johnnie Ray, and Jimmy Durante, thanks to her success as a recording artist.",
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
        # Same for overfit test
        row_test_inner(
            batch, question, sentences, no_paraphrase_question, paraphrased_questions
        )


if __name__ == "__main__":
    unittest.main()
