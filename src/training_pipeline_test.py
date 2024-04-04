import unittest
from training_pipeline import TrainingPipeline
from config import Config


# python -m unittest training_pipeline_test.TestTrainingPipeline -v
class TestTrainingPipeline(unittest.TestCase):
    def test_load_config(self):
        config_path = "experiments/t5base_mpnet_topk_2_flat_10.yaml"

        config = Config.from_file(config_path)

        # Should not crash
        TrainingPipeline(config)

    # python -m unittest training_pipeline_test.TestTrainingPipeline.test_train_one_epoch -v
    def test_train_one_epoch(self):
        config_path = "experiments/dummy_experiment.yaml"

        config = Config.from_file(config_path)

        pipeline = TrainingPipeline(config)

        for _ in range(10):
            loss = pipeline.train_one_epoch()
            print(loss)

    # python -m unittest training_pipeline_test.TestTrainingPipeline.test_run_validation -v
    def test_run_validation(self):
        config_path = "experiments/dummy_experiment.yaml"

        config = Config.from_file(config_path)

        pipeline = TrainingPipeline(config)
        metrics = pipeline.run_validation()
        print(metrics)


if __name__ == "__main__":
    unittest.main()
