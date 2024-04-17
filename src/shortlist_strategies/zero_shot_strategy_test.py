import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest shortlist_strategies.zero_shot_strategy_test.TestZeroShotStrategy -v
class TestZeroShotStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.zero_shot_strategy_test.TestZeroShotStrategy.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/zeroshot_test_experiment.yaml")
        config.offline_validation.type = "zero_shot"
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert indexes == [], indexes
        assert confidences == [], confidences

    # python -m unittest shortlist_strategies.zero_shot_strategy_test.TestZeroShotStrategy.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/zeroshot_test_experiment.yaml")
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

            assert len(few_shots["prompts"]) == 0, few_shots
            assert len(few_shots["labels"]) == 0, few_shots
