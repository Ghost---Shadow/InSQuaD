import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
import torch


# python -m unittest shortlist_strategies.fast_vote_k_test.TestFastVoteK -v
class TestFastVoteK(unittest.TestCase):

    # python -m unittest shortlist_strategies.fast_vote_k_test.TestFastVoteK.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/tests/fastvotek_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        indexes, confidences = pipeline.shortlist_strategy.shortlist()

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            7,
            92,
            210,
            18,
            223,
            177,
            135,
            131,
            22,
            206,
            29,
            299,
            47,
            77,
            64,
            204,
            151,
            165,
            152,
            286,
        ], indexes
        assert confidences == [
            268,
            49.90000000000018,
            10.929999999999966,
            2.4899999999999887,
            0.6180999999999994,
            0.11102000000000005,
            0.028296999999999982,
            0.007583300000000003,
            0.0019121799999999986,
            0.0005779600000000004,
            0.00012548409999999982,
            2.9745700000000026e-05,
            7.701588999999996e-06,
            1.4048827000000021e-06,
            3.3737423000000037e-07,
            8.130777300000001e-08,
            2.0425471200000007e-08,
            3.877177220000002e-09,
            1.139576822e-09,
            2.962041299999999e-10,
        ], confidences

    # python -m unittest shortlist_strategies.fast_vote_k_test.TestFastVoteK.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/fastvotek_test_experiment.yaml")
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
