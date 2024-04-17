import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest shortlist_strategies.random_strategy_test.TestRandomStrategy -v
class TestRandomStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.random_strategy_test.TestRandomStrategy.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.offline_validation.type = "random"
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert indexes == [
            179,
            2679,
            3133,
            1061,
            1091,
            2230,
            2287,
            149,
            965,
            450,
            32,
            1807,
            1465,
            109,
            415,
            3479,
            490,
            325,
            842,
            564,
        ], indexes
        assert confidences is None
