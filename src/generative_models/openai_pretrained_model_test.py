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

    # python -m unittest generative_models.openai_pretrained_model_test.TestWrappedOpenAiPretrained.test_evaluate -v
    def test_evaluate(self):
        config = Config.from_file(
            "experiments/model_size_ablations/zeroshot_mpnet_davinci2.yaml"
        )
        wrapped_model = WrappedOpenAiPretrained(
            config, config.offline_validation.generative_model
        )

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_model.evaluate(prompt, label1)
        result2 = wrapped_model.evaluate(prompt, label2)

        assert result1 == {
            "target_sequence_probability": -1,
            "predicted_sequence_probability": 0.6685865372033624,
            "target": " jumps over the lazy dog",
            "predicted": " jumps over the lazy dog",
        }, result1
        assert result2 == {
            "target_sequence_probability": -1,
            "predicted_sequence_probability": 0.6685865372033624,
            "target": " brick dig hat mat late",
            "predicted": " jumps over the lazy dog",
        }, result2
