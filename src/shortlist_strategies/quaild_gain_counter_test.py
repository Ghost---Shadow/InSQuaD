import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest shortlist_strategies.quaild_gain_counter_test.TestQuaildGainCounterStrategy -v
class TestQuaildGainCounterStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.quaild_gain_counter_test.TestQuaildGainCounterStrategy.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
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

    # python -m unittest shortlist_strategies.quaild_gain_counter_test.TestQuaildGainCounterStrategy.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)

        if not os.path.exists(pipeline.shortlisted_data_path):
            pipeline.shortlist("mrpc")

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
