import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline -v
class TestOfflineEvaluationPipeline(unittest.TestCase):
    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_quaild_flow -v
    def test_quaild_flow(self):
        config_path = "experiments/tests/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_random_flow -v
    def test_random_flow(self):
        config_path = "experiments/tests/random_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_zeroshot_flow -v
    def test_zeroshot_flow(self):
        config_path = "experiments/tests/zeroshot_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_least_confidence_flow -v
    def test_least_confidence_flow(self):
        config_path = "experiments/tests/leastconfidence_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_fast_vote_k_flow -v
    def test_fast_vote_k_flow(self):
        config_path = "experiments/tests/fastvotek_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_vote_k_flow -v
    def test_vote_k_flow(self):
        config_path = "experiments/tests/votek_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_ideal_flow -v
    def test_ideal_flow(self):
        config_path = "experiments/tests/ideal_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()


if __name__ == "__main__":
    unittest.main()
