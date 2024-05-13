import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest shortlist_strategies.oracle_strategy_test.TestOracleStrategy -v
class TestOracleStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.oracle_strategy_test.TestOracleStrategy.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/tests/oracle_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        indexes, confidences = pipeline.shortlist_strategy.shortlist()

        assert indexes == [
            203,
            266,
            152,
            9,
            233,
            226,
            196,
            109,
            5,
            175,
            237,
            57,
            218,
            45,
            182,
            221,
            289,
            211,
            148,
            165,
        ], indexes
        assert confidences == [
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
            0.0033333333333333335,
        ], confidences

    # python -m unittest shortlist_strategies.oracle_strategy_test.TestOracleStrategy.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/oracle_test_experiment.yaml")
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
