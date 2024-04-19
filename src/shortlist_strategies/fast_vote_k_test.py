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
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            214,
            149,
            42,
            196,
            171,
            112,
            106,
            246,
            152,
            61,
            63,
            183,
            247,
            43,
            91,
            108,
            169,
            46,
            229,
            107,
        ], indexes
        assert confidences == [
            230,
            42.20000000000015,
            8.83999999999996,
            1.6769999999999945,
            0.38699999999999946,
            0.08341000000000018,
            0.01677599999999995,
            0.004402200000000002,
            0.000950240000000001,
            0.00018937699999999974,
            3.802980000000006e-05,
            8.244829999999996e-06,
            1.7945210000000007e-06,
            4.01222100000001e-07,
            1.0102440999999986e-07,
            1.7243819999999988e-08,
            4.301763999999997e-09,
            8.954220999999998e-10,
            1.958012000000004e-10,
            3.933156900000005e-11,
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
