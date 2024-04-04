import unittest
from config import Config
import torch
from src.shortlist_strategies.quaild import QuaildShortlistStrategy
from train_utils import set_seed
from training_pipeline import TrainingPipeline


# python -m unittest training_strategies.quaild_test.TestQuaildShortlistStrategy -v
class TestQuaildShortlistStrategy(unittest.TestCase):
    # python -m unittest training_strategies.quaild_test.TestQuaildShortlistStrategy.test_shortlist -v
    def test_shortlist(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        pipeline = TrainingPipeline(config)
        strategy = QuaildShortlistStrategy(config, pipeline)
        shortlist = strategy.shortlist()

        assert len(shortlist) == config.validation.annotation_budget
