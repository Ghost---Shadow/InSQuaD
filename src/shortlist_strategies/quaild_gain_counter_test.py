import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
from train_utils import set_seed


# python -m unittest shortlist_strategies.quaild_gain_counter_test.TestQuaildGainCounterStrategy -v
class TestQuaildGainCounterStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.quaild_gain_counter_test.TestQuaildGainCounterStrategy.test_shortlist_dummy -v
    def test_shortlist_dummy(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert indexes == [
            2846,
            3506,
            3034,
            2044,
            608,
            2619,
            818,
            1646,
            2997,
            100,
            2013,
            527,
            2396,
            1001,
            2552,
            3142,
            2197,
            2025,
            3335,
            2359,
        ], indexes
        assert confidences == [
            1.0,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
        ], confidences
