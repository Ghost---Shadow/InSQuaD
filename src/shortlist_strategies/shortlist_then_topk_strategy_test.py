import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest shortlist_strategies.shortlist_then_topk_strategy_test.ShortlistThenTopK -v
class ShortlistThenTopK(unittest.TestCase):
    # python -m unittest shortlist_strategies.shortlist_then_topk_strategy_test.ShortlistThenTopK.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file(
            "experiments/tests/shortlistandtopk_test_experiment.yaml"
        )
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        indexes, confidences = pipeline.shortlist_strategy.shortlist()

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            79,
            233,
            216,
            196,
            197,
            66,
            183,
            128,
            297,
            163,
            120,
            230,
            171,
            267,
            223,
            175,
            201,
            57,
            243,
            84,
        ], indexes
        assert confidences == [
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
        ], confidences

    # python -m unittest shortlist_strategies.shortlist_then_topk_strategy_test.ShortlistThenTopK.test_oom -v
    def test_oom(self):
        config = Config.from_file(
            "experiments/tests/shortlistandtopk_test_experiment.yaml"
        )
        config.offline_validation.subsample_for_train_size = 3000
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not OOM
        pipeline.shortlist_strategy.shortlist()

    # python -m unittest shortlist_strategies.shortlist_then_topk_strategy_test.ShortlistThenTopK.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file(
            "experiments/tests/shortlistandtopk_test_experiment.yaml"
        )
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        if not os.path.exists(pipeline.shortlisted_data_path):
            pipeline.shortlist()

        for row, few_shots in pipeline.shortlist_strategy.assemble_few_shot(
            use_cache=False
        ):
            assert "prompts" in row, row
            assert "labels" in row, row

            assert "prompts" in few_shots, few_shots
            assert "labels" in few_shots, few_shots

            assert (
                len(few_shots["prompts"]) == config.offline_validation.num_shots
            ), few_shots
            assert (
                len(few_shots["labels"]) == config.offline_validation.num_shots
            ), few_shots
