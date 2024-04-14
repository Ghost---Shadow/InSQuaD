import unittest
from config import Config
from generative_models.wrapped_t5 import WrappedT5


# python -m unittest generative_models.wrapped_t5_test.TestWrappedT5 -v
class TestWrappedT5(unittest.TestCase):

    def test_evaluate(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        wrapped_t5 = WrappedT5(config, config.offline_validation.generative_model)

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_t5.evaluate(prompt, label1)
        result2 = wrapped_t5.evaluate(prompt, label2)

        assert result1["sequence_probability"] > result2["sequence_probability"]
        assert result1["actual"] == "Thes up", result1["actual"]
        assert result2["actual"] == "The wallsfoxx", result2["actual"]


if __name__ == "__main__":
    unittest.main()
