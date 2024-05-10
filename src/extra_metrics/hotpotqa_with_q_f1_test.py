import unittest
from config import Config
from extra_metrics.hotpotqa_with_q_f1 import ExtraMetricHotpotQaWithQF1
from training_pipeline import TrainingPipeline


# python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1 -v
class TestExtraMetricHotpotQaWithQF1(unittest.TestCase):
    # python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1.test_all_correct_no_paraphrases -v
    def test_all_correct_no_paraphrases(self):
        """All predicted indices are correct and no paraphrases are involved."""
        predicted_indices = [1, 2, 3]
        no_paraphrase_idxs = [1, 2, 3]
        paraphrase_lut = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}
        self.assertEqual(
            ExtraMetricHotpotQaWithQF1._count_actually_correct(
                predicted_indices, no_paraphrase_idxs, paraphrase_lut
            ),
            3,
        )

    # python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1.test_some_correct_with_paraphrases -v
    def test_some_correct_with_paraphrases(self):
        """Some predicted indices are correct, including paraphrases."""
        predicted_indices = [1, 5, 3]
        no_paraphrase_idxs = [1, 2, 3]
        paraphrase_lut = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}
        self.assertEqual(
            ExtraMetricHotpotQaWithQF1._count_actually_correct(
                predicted_indices, no_paraphrase_idxs, paraphrase_lut
            ),
            3,
        )

    # python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1.test_duplicates_and_paraphrases -v
    def test_duplicates_and_paraphrases(self):
        """Predicted indices contain duplicates and paraphrases."""
        predicted_indices = [1, 5, 5, 3, 6]
        no_paraphrase_idxs = [1, 2, 3]
        paraphrase_lut = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}
        self.assertEqual(
            ExtraMetricHotpotQaWithQF1._count_actually_correct(
                predicted_indices, no_paraphrase_idxs, paraphrase_lut
            ),
            3,
        )

    # python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1.test_no_correct_predictions -v
    def test_no_correct_predictions(self):
        """No correct predictions."""
        predicted_indices = [4, 5, 6]
        no_paraphrase_idxs = [1, 2, 3]
        paraphrase_lut = {4: 44, 5: 55, 6: 66}
        self.assertEqual(
            ExtraMetricHotpotQaWithQF1._count_actually_correct(
                predicted_indices, no_paraphrase_idxs, paraphrase_lut
            ),
            0,
        )

    # python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1.test_all_correct_with_all_paraphrases -v
    def test_all_correct_with_all_paraphrases(self):
        """All predicted indices are correct but are all paraphrases."""
        predicted_indices = [4, 5, 6]
        no_paraphrase_idxs = [1, 2, 3]
        paraphrase_lut = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}
        self.assertEqual(
            ExtraMetricHotpotQaWithQF1._count_actually_correct(
                predicted_indices, no_paraphrase_idxs, paraphrase_lut
            ),
            3,
        )

    # python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1.test_generate_metric -v
    def test_generate_metric(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = TrainingPipeline(config)

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        extra_metric = ExtraMetricHotpotQaWithQF1(pipeline)
        metrics = extra_metric.generate_metric(batch)

        assert metrics == {
            "precision": 0.16666666666666666,
            "recall": 0.75,
            "f1_score": 0.27272727272727276,
        }, metrics

    # python -m unittest extra_metrics.hotpotqa_with_q_f1_test.TestExtraMetricHotpotQaWithQF1.test_sweeping_pr_curve -v
    def test_sweeping_pr_curve(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = TrainingPipeline(config)

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        extra_metric = ExtraMetricHotpotQaWithQF1(pipeline)
        result_score, result_k = extra_metric.sweeping_pr_curve(batch)

        assert result_score == {
            "-0.0153": [0.11111111111111112],
            "-0.0051": [0.11111111111111112],
            "-0.0057": [0.22222222222222224],
            "-0.0042": [0.33333333333333326],
            "-0.0030": [0.33333333333333326],
            "-0.0026": [0.33333333333333326],
            "-0.0021": [0.33333333333333326],
            "-0.0019": [0.33333333333333326],
        }, result_score

        assert result_k == {
            1: [0.11111111111111112],
            2: [0.11111111111111112],
            3: [0.22222222222222224],
            4: [0.33333333333333326],
            5: [0.33333333333333326],
            6: [0.33333333333333326],
            7: [0.33333333333333326],
            8: [0.33333333333333326],
        }, result_k


if __name__ == "__main__":
    unittest.main()
