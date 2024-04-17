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
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=False)

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_generate_few_shots -v
    def test_generate_few_shots(self):
        config_path = "experiments/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist()
        pipeline.generate_few_shots(skip_if_done=False)

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_generate_run_inference -v
    def test_generate_run_inference(self):
        config_path = "experiments/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist()
        pipeline.generate_few_shots()
        pipeline.run_inference(skip_if_done=False)

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_random_flow -v
    def test_random_flow(self):
        config_path = "experiments/random_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)


if __name__ == "__main__":
    unittest.main()
