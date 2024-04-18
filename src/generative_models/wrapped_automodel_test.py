import unittest
from config import Config
from dataloaders.dbpedia import DBPedia
from generative_models.wrapped_automodel import WrappedAutoModel


# python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel -v
class TestWrappedAutoModel(unittest.TestCase):

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_single_token -v
    def test_single_token(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = (
            "EleutherAI/gpt-neo-125m"
        )
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = "I have a"
        label = " dog"

        result = wrapped_model.evaluate(prompt, label)

        assert result == {
            "target_sequence_probability": 0.000470790226245299,
            "predicted_sequence_probability": 0.03734087198972702,
            "target": " dog",
            "predicted": " problem",
        }, result

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_stablelm -v
    def test_evaluate_stablelm(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
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
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
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
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
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

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_gpt2 -v
    def test_evaluate_gpt2(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = "openai-community/gpt2"
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

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_with_options_neo175m -v
    def test_evaluate_with_options_neo175m(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = (
            "EleutherAI/gpt-neo-125m"
        )
        wrapped_model = WrappedAutoModel(
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
                "yes": 0.9760239720344543,
                "no": 0.21766287088394165,
            },
            "correct": True,
        }, result

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_lower_than_max_tokens -v
    def test_lower_than_max_tokens(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = (
            "stabilityai/stablelm-2-1_6b"
        )
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = """Q: Title: Henkel
Content:  Henkel AG & Company KGaA operates worldwide with leading brands and technologies in three business areas: Laundry & Home Care Beauty Care and Adhesive Technologies. Henkel is the name behind some of America’s favorite brands.
Topic:
A: """
        options = list(DBPedia.LABELS.values())
        correct_option_index = 0

        result = wrapped_model.evaluate_with_options(
            prompt, correct_option_index, options
        )

        assert result == {
            "option_probabilities": {
                "Company": 0.9973655939102173,
                "EducationalInstitution": 0.0030768769793212414,
                "Artist": 0.0007606353610754013,
                "Athlete": 0.0005123738665133715,
                "OfficeHolder": 0.062278542667627335,
                "MeanOfTransportation": 0.0010649517644196749,
                "Building": 0.02669544331729412,
                "NaturalPlace": 0.0023292030673474073,
                "Village": 0.01895320601761341,
                "Animal": 0.007577642798423767,
                "Plant": 0.004710435401648283,
                "Album": 0.005166991148144007,
                "Film": 0.0006463695899583399,
                "WrittenWork": 0.013707134872674942,
            },
            "correct": True,
        }, result


if __name__ == "__main__":
    unittest.main()
