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

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_quaild_flow_xsum -v
    def test_quaild_flow_xsum(self):
        config_path = "experiments/tests/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)
        config.offline_validation.datasets = ["xsum"]
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "xsum"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_quaild_flow_mwoz -v
    def test_quaild_flow_mwoz(self):
        config_path = "experiments/tests/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)
        config.offline_validation.datasets = ["mwoz"]
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mwoz"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_quaild_flow_geoq -v
    def test_quaild_flow_geoq(self):
        config_path = "experiments/tests/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)
        config.offline_validation.datasets = ["geoq"]
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "geoq"

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

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_insquad_ld_flow -v
    def test_insquad_ld_flow(self):
        config_path = "experiments/tests/insquad_ld_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_quaild_similar_flow -v
    def test_quaild_similar_flow(self):
        config_path = "experiments/tests/quaild_test_similar.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_mfl_flow -v
    def test_mfl_flow(self):
        config_path = "experiments/tests/mfl_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_gc_flow -v
    def test_gc_flow(self):
        config_path = "experiments/tests/gc_test_experiment.yaml"

        config = Config.from_file(config_path)
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not crash
        pipeline.shortlist(skip_if_done=True)
        pipeline.generate_few_shots(skip_if_done=True)
        pipeline.run_inference(skip_if_done=True)
        pipeline.analyze_inference_outputs()

    # python -m unittest offline_eval_pipeline_test.TestOfflineEvaluationPipeline.test_diversity_flow -v
    def test_diversity_flow(self):
        config_path = "experiments/tests/diversity_test_experiment.yaml"

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
