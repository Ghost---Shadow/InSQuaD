import torch
from config import Config
from training_pipeline import TrainingPipeline
from subset_selection_strategies.flat_cutoff import FlatCutoffStrategy
import unittest


# python -m unittest subset_selection_strategies.flat_cutoff_test.TestFlatCutoffStrategy -v
class TestFlatCutoffStrategy(unittest.TestCase):
    # python -m unittest subset_selection_strategies.flat_cutoff_test.TestFlatCutoffStrategy.test_subset_select -v
    def test_subset_select(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"
        config.architecture.dense_index.type = "in_memory"
        config.architecture.subset_selection_strategy.type = "flat_cutoff"
        config.offline_validation.datasets = []  # Save time
        pipeline = TrainingPipeline(config)

        # Create an instance of the strategy
        strategy = FlatCutoffStrategy(config, pipeline)

        query_embedding = torch.tensor([1, 1, 0], dtype=torch.float32)
        shortlist_embeddings = torch.tensor(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )

        # Apply the indexes
        result, scores = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = [1, 0, 2]
        expected_scores = [2.0, 1.0, 1.0]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()
