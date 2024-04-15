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

        # Result 1
        assert (
            result1["target_sequence_probability"]
            == result1["predicted_sequence_probability"]
        )
        assert result1["predicted"] == result1["target"]
        assert result1["predicted"] == " jumps over the lazy dog", (
            "(" + result1["predicted"] + ")"
        )

        # Result 2
        assert result2["predicted"] == " jumps over the lazy dog", (
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

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_gemma -v
    def test_evaluate_gemma(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = "google/gemma-2b"
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_model.evaluate(prompt, label1)
        result2 = wrapped_model.evaluate(prompt, label2)

        # Result 1
        assert (
            result1["target_sequence_probability"]
            == result1["predicted_sequence_probability"]
        )
        assert result1["predicted"] == result1["target"]
        assert result1["predicted"] == " jumps over the lazy dog", (
            "(" + result1["predicted"] + ")"
        )

        # Result 2
        assert result2["predicted"] == " jumps over the lazy dog", (
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

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_neo175m -v
    def test_evaluate_neo175m(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = (
            "EleutherAI/gpt-neo-125m"
        )
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_model.evaluate(prompt, label1)
        result2 = wrapped_model.evaluate(prompt, label2)

        # print(result1)
        # print(result2)

        # Result 1
        assert result1["predicted"] == "es are a great way", (
            "(" + result1["predicted"] + ")"
        )

        # Result 2
        assert result2["predicted"] == "es are a great way", (
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
