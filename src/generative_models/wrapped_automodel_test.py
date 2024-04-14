import unittest
from config import Config
from generative_models.wrapped_automodel import WrappedAutoModel


# python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel -v
class TestWrappedAutoModel(unittest.TestCase):

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_stablelm -v
    def test_evaluate_stablelm(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = (
            "stabilityai/stablelm-2-1_6b"
        )
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_model.evaluate(prompt, label1)
        result2 = wrapped_model.evaluate(prompt, label2)

        assert result1["sequence_probability"] > result2["sequence_probability"]
        assert result1["actual"] == "  and fox jumps", "(" + result1["actual"] + ")"
        assert result2["actual"] == "  and fox jumps", "(" + result2["actual"] + ")"


if __name__ == "__main__":
    unittest.main()
