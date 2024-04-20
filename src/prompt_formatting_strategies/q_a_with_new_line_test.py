import json
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
from prompt_formatting_strategies.q_a_with_new_line import QAWithNewLine
from train_utils import set_seed
from transformers import T5Tokenizer


# python -m unittest prompt_formatting_strategies.q_a_with_new_line_test.TestQAWithNewLine -v
class TestQAWithNewLine(unittest.TestCase):
    # python -m unittest prompt_formatting_strategies.q_a_with_new_line_test.TestQAWithNewLine.test_happy_path -v
    def test_happy_path(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        tokenizer = pipeline.generative_model.tokenizer
        formatter = pipeline.prompt_formatting_strategy

        batch = {
            "prompts": "What is Alice's favourite fruit?",
            "labels": "apple",
        }

        few_shots = {
            "prompts": [
                "What is Bob's favourite fruit?",
                "What is Charlie's favourite fruit?",
            ],
            "labels": ["banana", "coconut"],
            "distances": [0.5622506737709045, 0.5896707773208618],
        }

        expected = "Q: What is Bob's favourite fruit?\nA: banana\n\nQ: What is Charlie's favourite fruit?\nA: coconut\n\nQ: What is Alice's favourite fruit?\nA: "

        actual = formatter.generate_prompt(tokenizer, batch, few_shots)

        assert actual == expected, json.dumps(actual)

    # python -m unittest prompt_formatting_strategies.q_a_with_new_line_test.TestQAWithNewLine.test_insufficient_context_length -v
    def test_insufficient_context_length(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        tokenizer = pipeline.generative_model.tokenizer
        formatter = pipeline.prompt_formatting_strategy
        tokenizer.model_max_length = 30  # Decrease max length

        batch = {
            "prompts": "What is Alice's favourite fruit?",
            "labels": "apple",
        }

        few_shots = {
            "prompts": [
                "What is Bob's favourite fruit?",
                "What is Charlie's favourite fruit?",
            ],
            "labels": ["banana", "coconut"],
            "distances": [0.5622506737709045, 0.5896707773208618],
        }

        expected = "Q: What is Bob's favourite fruit?\nA: banana\n\nQ: What is Alice's favourite fruit?\nA: "

        formatter = QAWithNewLine(config)

        actual = formatter.generate_prompt(tokenizer, batch, few_shots)

        assert actual == expected, json.dumps(actual)
