import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
import torch


# python -m unittest shortlist_strategies.ideal_test.TestIdeal -v
class TestIdeal(unittest.TestCase):

    # python -m unittest shortlist_strategies.ideal_test.TestIdeal.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/tests/ideal_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            0,
            115,
            251,
            193,
            191,
            217,
            1,
            179,
            243,
            56,
            67,
            154,
            113,
            20,
            12,
            66,
            150,
            35,
            30,
            8,
        ], indexes
        assert confidences == [0] * len(indexes), confidences

    # python -m unittest shortlist_strategies.ideal_test.TestIdeal.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/ideal_test_experiment.yaml")
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
