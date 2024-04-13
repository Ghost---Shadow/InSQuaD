import unittest

from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline -v
class TestOfflineEvaluationPipeline(unittest.TestCase):
    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_shortlist -v
    def test_shortlist(self):
        config_path = "experiments/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)

        # Should not crash
        pipeline.shortlist("mrpc")


if __name__ == "__main__":
    unittest.main()
