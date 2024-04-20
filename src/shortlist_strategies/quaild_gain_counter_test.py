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
            175,
            93,
            43,
            34,
            206,
            53,
            27,
            289,
            16,
            138,
            2,
            210,
            4,
            234,
            110,
            72,
            216,
            285,
            224,
            79,
        ], indexes
        assert confidences == [
            1.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
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
