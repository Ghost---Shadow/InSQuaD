import torch
from config import Config
from training_pipeline import TrainingPipeline
from subset_selection_strategies.gain_cutoff import GainCutoffStrategy
import unittest


# python -m unittest subset_selection_strategies.base_strategy_test.TestGainCutoffStrategy -v
class TestGainCutoffStrategy(unittest.TestCase):
    def subset_select(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        pipeline = TrainingPipeline(config)

        # Create an instance of the strategy
        strategy = GainCutoffStrategy(config, pipeline)

        query_embedding = torch.Tensor([1, 0, 0], dtype=torch.float32)
        shortlist_embeddings = torch.Tensor(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=torch.float32,
        )

        # Apply the indexes
        result = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = torch.Tensor([0])

        # Assert that the result matches the expected output
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
