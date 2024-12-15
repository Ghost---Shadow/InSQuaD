import torch
from config import Config
from training_pipeline import TrainingPipeline
from subset_selection_strategies.maximum_facility_location import (
    MaximumFacilityLocationSubsetStrategy,
)
import unittest
import torch.nn.functional as F


# python -m unittest subset_selection_strategies.maximum_facility_location_test.TestMaximumFacilityLocationSubsetStrategy -v
class TestMaximumFacilityLocationSubsetStrategy(unittest.TestCase):
    # python -m unittest subset_selection_strategies.maximum_facility_location_test.TestMaximumFacilityLocationSubsetStrategy.test_subset_select -v
    def test_subset_select(self):
        config = Config.from_file("experiments/tests/mfl_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"
        config.architecture.dense_index.type = "in_memory"
        config.offline_validation.datasets = []  # Save time
        config.architecture.subset_selection_strategy.k = 5
        config.offline_validation.annotation_budget = 5
        # config.offline_validation.q_d_tradeoff_lambda = 0.5
        # config.architecture.subset_selection_strategy.gain_cutoff = 0.0
        pipeline = TrainingPipeline(config)

        shortlist_embeddings = torch.tensor(
            [
                [1.0000, 0.0000, 0.0000],
                [0.7071, 0.7071, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.7071, 0.7071, 0.0000],
                [-0.7071, -0.7071, 0.0000],
            ],
            dtype=torch.float32,
        )

        shortlist_embeddings = F.normalize(shortlist_embeddings, dim=-1)

        # Apply the indexes
        result, scores = pipeline.subset_selection_strategy.subset_select(
            shortlist_embeddings
        )

        expected_output = [1, 2, 4, 0, 3]
        expected_scores = [
            1.7071069478988647,
            3.7071070671081543,
            4.707107067108154,
            5.0,
            5.0,
        ]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.maximum_facility_location_test.TestMaximumFacilityLocationSubsetStrategy.test_real_data -v
    def test_real_data(self):
        config = Config.from_file("experiments/tests/mfl_test_experiment.yaml")
        pipeline = TrainingPipeline(config)
        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        metrics = pipeline.compute_extra_metrics(batch)

        # assert metrics == {
        #     "precision": 1.0,
        #     "recall": 0.125,
        #     "f1_score": 0.2222222222222222,
        # }, metrics

        assert metrics == {
            "precision": 0.15,
            "recall": 0.375,
            "f1_score": 0.21428571428571425,
        }, metrics

    # python -m unittest subset_selection_strategies.maximum_facility_location_test.TestMaximumFacilityLocationSubsetStrategy.test_shorter_than_budget -v
    def test_shorter_than_budget(self):
        config = Config.from_file("experiments/tests/mfl_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"  # Save time
        config.architecture.dense_index.type = "in_memory"  # Save time
        config.offline_validation.datasets = []  # Save time
        pipeline = TrainingPipeline(config)

        # Create an instance of the strategy
        strategy = MaximumFacilityLocationSubsetStrategy(config, pipeline)

        query_embedding = None
        shortlist_embeddings = torch.tensor(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=torch.float32,
        )

        # Apply the indexes
        result, scores = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = [0, 2, 1]
        expected_scores = [2.0, 3.0, 3.0]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.maximum_facility_location_test.TestMaximumFacilityLocationSubsetStrategy.test_empty_shortlist -v
    def test_empty_shortlist(self):
        config = Config.from_file("experiments/tests/mfl_test_experiment.yaml")
        config.offline_validation.datasets = []
        pipeline = TrainingPipeline(config)

        strategy = MaximumFacilityLocationSubsetStrategy(config, pipeline)

        query_embedding = None
        shortlist_embeddings = torch.tensor([], dtype=torch.float32).reshape(0, 3)

        result, scores = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = []
        expected_scores = []

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()


if __name__ == "__main__":
    unittest.main()
