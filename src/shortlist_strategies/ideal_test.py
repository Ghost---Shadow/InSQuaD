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
        pipeline.current_dataset_name = "mrpc"

        indexes, scores = pipeline.shortlist_strategy.shortlist()

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            0,
            58,
            227,
            28,
            48,
            179,
            263,
            102,
            157,
            30,
            259,
            21,
            217,
            289,
            94,
            40,
            238,
            88,
            220,
            145,
        ], indexes
        assert scores == [
            0,
            22.7,
            32.1,
            39.4,
            41.3,
            49.8,
            50.9,
            55.6,
            57.2,
            62.9,
            60.5,
            66.6,
            74.2,
            75.6,
            73.1,
            81.1,
            81.6,
            82.0,
            84.3,
            90.8,
        ], scores

    # python -m unittest shortlist_strategies.ideal_test.TestIdeal.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/ideal_test_experiment.yaml")
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
