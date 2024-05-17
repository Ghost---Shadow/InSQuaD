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
            "rouge": {
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            },
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

        assert min(wrapped_model.tokenizer.max_model_input_sizes.values()) == 1024, min(
            wrapped_model.tokenizer.max_model_input_sizes.values()
        )

        assert (
            wrapped_model.model.config.max_position_embeddings == 4096
        ), wrapped_model.model.config.max_position_embeddings

        assert (
            wrapped_model.tokenizer.model_max_length == 1024
        ), wrapped_model.tokenizer.model_max_length

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_model.evaluate(prompt, label1)
        result2 = wrapped_model.evaluate(prompt, label2)

        assert result1 == {
            "target_sequence_probability": 0.34326171875,
            "predicted_sequence_probability": 0.34326171875,
            "target": " jumps over the lazy dog",
            "predicted": " jumps over the lazy dog",
            "rouge": {
                "rouge1": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
                "rouge2": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
                "rougeL": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
            },
        }, result1
        assert result2 == {
            "target_sequence_probability": 0.0,
            "predicted_sequence_probability": 0.34326171875,
            "target": " brick dig hat mat late",
            "predicted": " jumps over the lazy dog",
            "rouge": {
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            },
        }, result2

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

        assert (
            list(wrapped_model.tokenizer.max_model_input_sizes.values()) == []
        ), wrapped_model.tokenizer.max_model_input_sizes.values()

        assert (
            wrapped_model.model.config.max_position_embeddings == 8192
        ), wrapped_model.model.config.max_position_embeddings

        assert (
            wrapped_model.tokenizer.model_max_length == 8192
        ), wrapped_model.tokenizer.model_max_length

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_model.evaluate(prompt, label1)
        result2 = wrapped_model.evaluate(prompt, label2)

        assert result1 == {
            "target_sequence_probability": 0.5524634718894958,
            "predicted_sequence_probability": 0.5524634718894958,
            "target": " jumps over the lazy dog",
            "predicted": " jumps over the lazy dog",
            "rouge": {
                "rouge1": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
                "rouge2": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
                "rougeL": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
            },
        }, result1
        assert result2 == {
            "target_sequence_probability": 2.5972147278655194e-33,
            "predicted_sequence_probability": 0.5524634718894958,
            "target": " brick dig hat mat late",
            "predicted": " jumps over the lazy dog",
            "rouge": {
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            },
        }, result2

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

        assert min(wrapped_model.tokenizer.max_model_input_sizes.values()) == 1024, min(
            wrapped_model.tokenizer.max_model_input_sizes.values()
        )

        assert (
            wrapped_model.model.config.max_position_embeddings == 2048
        ), wrapped_model.model.config.max_position_embeddings

        assert (
            wrapped_model.tokenizer.model_max_length == 1024
        ), wrapped_model.tokenizer.model_max_length

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_model.evaluate(prompt, label1)
        result2 = wrapped_model.evaluate(prompt, label2)

        assert result1 == {
            "target_sequence_probability": 4.3222193944585097e-13,
            "predicted_sequence_probability": 5.4240999816101976e-06,
            "target": " jumps over the lazy dog",
            "predicted": "es are a great way",
            "rouge": {
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            },
        }, result1
        assert result2 == {
            "target_sequence_probability": 6.99859156647953e-23,
            "predicted_sequence_probability": 5.4240999816101976e-06,
            "target": " brick dig hat mat late",
            "predicted": "es are a great way",
            "rouge": {
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            },
        }, result2

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

        prompt = "Q: Do you want a banana? yes or no?"
        options = ["yes", "no"]
        correct_option_index = 0

        result = wrapped_model.evaluate_with_options(
            prompt, correct_option_index, options
        )

        assert result == {
            "option_probabilities": {
                "yes": 0.9760242104530334,
                "no": 0.21766166388988495,
            },
            "correct": True,
        }, result

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_with_options_gptj6b -v
    def test_evaluate_with_options_gptj6b(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = (
            # "ainize/gpt-j-6B-float16" # OOM
            "ainize/gpt-j-6B-float16"
        )
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = "Q: Do you want a banana? yes or no?"
        options = ["yes", "no"]
        correct_option_index = 0

        result = wrapped_model.evaluate_with_options(
            prompt, correct_option_index, options
        )

        assert result == {
            "option_probabilities": {
                "yes": 0.9760242104530334,
                "no": 0.21766166388988495,
            },
            "correct": True,
        }, result

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_with_options_gemma7b -v
    def test_evaluate_with_options_gemma7b(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = "google/gemma-7b"
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = "Q: Do you want a banana? yes or no?"
        options = ["yes", "no"]
        correct_option_index = 0

        result = wrapped_model.evaluate_with_options(
            prompt, correct_option_index, options
        )

        assert result == {
            "option_probabilities": {
                "yes": 0.9385432600975037,
                "no": 0.3452707529067993,
            },
            "correct": True,
        }, result

    # python -m unittest generative_models.wrapped_automodel_test.TestWrappedAutoModel.test_evaluate_with_options_llama7b -v
    def test_evaluate_with_options_llama7b(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.generative_model.checkpoint = (
            "meta-llama/Llama-2-7b-hf"
        )
        wrapped_model = WrappedAutoModel(
            config, config.offline_validation.generative_model
        )

        prompt = "Q: Do you want a banana? yes or no?"
        options = ["yes", "no"]
        correct_option_index = 0

        result = wrapped_model.evaluate_with_options(
            prompt, correct_option_index, options
        )

        # Llama seems bugged
        assert result == {
            "option_probabilities": {"yes": float("nan"), "no": float("nan")},
            "correct": False,
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
