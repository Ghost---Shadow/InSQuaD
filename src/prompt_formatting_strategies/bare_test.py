import json
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
from prompt_formatting_strategies.bare import BareStrategy


# python -m unittest prompt_formatting_strategies.bare_test.TestBareStrategy -v
class TestBareStrategy(unittest.TestCase):
    # python -m unittest prompt_formatting_strategies.bare_test.TestBareStrategy.test_happy_path -v
    def test_happy_path(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.prompt_formatting_strategy.type = BareStrategy.NAME
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

        expected = "What is Bob's favourite fruit?bananaWhat is Charlie's favourite fruit?coconutWhat is Alice's favourite fruit?"

        actual = formatter.generate_prompt(tokenizer, batch, few_shots)

        assert actual == expected, json.dumps(actual)

    # python -m unittest prompt_formatting_strategies.bare_test.TestBareStrategy.test_insufficient_context_length -v
    def test_insufficient_context_length(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.prompt_formatting_strategy.type = BareStrategy.NAME
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        tokenizer = pipeline.generative_model.tokenizer
        tokenizer.model_max_length = 30  # Decrease max length
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

        expected = "What is Bob's favourite fruit?bananaWhat is Charlie's favourite fruit?coconutWhat is Alice's favourite fruit?"

        actual = formatter.generate_prompt(tokenizer, batch, few_shots)

        assert actual == expected, json.dumps(actual)

    # python -m unittest prompt_formatting_strategies.bare_test.TestBareStrategy.test_insufficient_context_length_for_prompt -v
    def test_insufficient_context_length_for_prompt(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.prompt_formatting_strategy.type = BareStrategy.NAME
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

        tokenizer.model_max_length = 3  # Decrease max length
        expected = "What is"
        actual = formatter.generate_prompt(tokenizer, batch, few_shots)
        assert actual == expected, json.dumps(actual)

        tokenizer.model_max_length = 8  # Decrease max length
        expected = "What is Alice's favourite fruit?"
        actual = formatter.generate_prompt(tokenizer, batch, few_shots)
        assert actual == expected, json.dumps(actual)
