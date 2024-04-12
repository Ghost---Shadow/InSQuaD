import unittest
from config import Config
import torch
from src.shortlist_strategies.quaild_gain_counter import QuaildGainCounterStrategy
from train_utils import set_seed
from training_pipeline import TrainingPipeline


# python -m unittest shortlist_strategies.quaild_test.TestQuaildGainCounterStrategy -v
class TestQuaildGainCounterStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.quaild_test.TestQuaildGainCounterStrategy.test_shortlist_dummy -v
    def test_shortlist_dummy(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.validation.datasets = ["dummy"]
        config.validation.annotation_budget = 2
        pipeline = TrainingPipeline(config)
        strategy = QuaildGainCounterStrategy(config, pipeline)
        shortlist = strategy.shortlist()

        print(shortlist)
