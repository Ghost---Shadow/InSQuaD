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
            "precision": 1.0,
            "recall": 0.125,
            "f1_score": 0.2222222222222222,
        }, metrics


if __name__ == "__main__":
    unittest.main()
