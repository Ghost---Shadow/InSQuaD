import unittest
from config import Config
from generative_models.openai_pretrained_model import WrappedOpenAiPretrained


# python -m unittest generative_models.openai_pretrained_model_test.TestWrappedOpenAiPretrained -v
@unittest.skip("Needs API")
class TestWrappedOpenAiPretrained(unittest.TestCase):
    # python -m unittest generative_models.openai_pretrained_model_test.TestWrappedOpenAiPretrained.test_evaluate_with_options -v
    def test_evaluate_with_options(self):
        config = Config.from_file(
            "experiments/model_size_ablations/zeroshot_mpnet_davinci2.yaml"
        )
        wrapped_model = WrappedOpenAiPretrained(
            config, config.offline_validation.generative_model
        )

        prompt = "Q: Do you want a banana? Yes or no?"
        options = ["yes", "no"]
        correct_option_index = 0

        result = wrapped_model.evaluate_with_options(
            prompt, correct_option_index, options
        )

        assert result == {
            "option_probabilities": {
                "yes": 0.6730636680301715,
                "no": 0.3269363319698285,
            },
            "correct": True,
        }, result
