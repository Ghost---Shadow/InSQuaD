import unittest
from config import Config
from generative_models.wrapped_t5 import WrappedT5


# python -m unittest generative_models.wrapped_t5_test.TestWrappedT5 -v
class TestWrappedT5(unittest.TestCase):

    def test_evaluate(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment_t5.yaml")
        wrapped_t5 = WrappedT5(config, config.offline_validation.generative_model)

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_t5.evaluate(prompt, label1)
        result2 = wrapped_t5.evaluate(prompt, label2)

        # Result 1
        assert result1["predicted"] == "The quick brown fox", (
            "(" + result1["predicted"] + ")"
        )

        # Result 2
        assert result2["predicted"] == "The quick brown fox", (
            "(" + result2["predicted"] + ")"
        )

        # Interaction
        assert (
            result1["target_sequence_probability"]
            > result2["target_sequence_probability"]
        ), (
            result1["target_sequence_probability"],
            result2["target_sequence_probability"],
        )


if __name__ == "__main__":
    unittest.main()
