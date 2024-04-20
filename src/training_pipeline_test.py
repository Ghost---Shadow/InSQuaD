import unittest
from training_pipeline import TrainingPipeline
from config import Config


# python -m unittest training_pipeline_test.TestTrainingPipeline -v
class TestTrainingPipeline(unittest.TestCase):
    # python -m unittest training_pipeline_test.TestTrainingPipeline.test_train_one_epoch -v
    def test_train_one_epoch(self):
        config_path = "experiments/tests/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)

        pipeline = TrainingPipeline(config)

        for _ in range(10):
            pipeline.train_one_epoch()

    # python -m unittest training_pipeline_test.TestTrainingPipeline.test_run_validation -v
    def test_run_validation(self):
        config_path = "experiments/tests/quaild_test_experiment.yaml"

        config = Config.from_file(config_path)

        pipeline = TrainingPipeline(config)
        metrics = pipeline.run_online_validation()
        print(metrics)


if __name__ == "__main__":
    unittest.main()
