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
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            228,
            6,
            79,
            206,
            117,
            185,
            242,
            167,
            9,
            30,
            180,
            222,
            230,
            217,
            136,
            68,
            199,
            15,
            96,
            24,
        ], indexes
        assert confidences == [
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
            0.00390625,
        ], confidences

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
            "mrpc", use_cache=False
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
