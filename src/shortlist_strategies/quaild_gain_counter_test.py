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
        pipeline.current_dataset_name = "mrpc"
        indexes, confidences = pipeline.shortlist_strategy.shortlist()

        assert indexes == [
            184,
            299,
            35,
            36,
            30,
            230,
            34,
            215,
            70,
            175,
            14,
            131,
            37,
            150,
            136,
            207,
            180,
            119,
            42,
            127,
        ], indexes
        assert confidences == [
            1.0,
            0.8333333333333334,
            0.8333333333333334,
            0.8333333333333334,
            0.8333333333333334,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.5,
            0.5,
            0.5,
            0.5,
        ], confidences

    # python -m unittest shortlist_strategies.quaild_gain_counter_test.TestQuaildGainCounterStrategy.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
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
