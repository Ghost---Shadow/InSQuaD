import torch
from config import Config
from training_pipeline import TrainingPipeline
from subset_selection_strategies.quaild_submodular import QuaildSubmodularStrategy
import unittest
import torch.nn.functional as F


# python -m unittest subset_selection_strategies.quaild_submodular_test.TestQuaildSubmodularStrategy -v
class TestQuaildSubmodularStrategy(unittest.TestCase):
    # python -m unittest subset_selection_strategies.quaild_submodular_test.TestQuaildSubmodularStrategy.test_subset_select -v
    def test_subset_select(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"
        config.architecture.dense_index.type = "in_memory"
        config.offline_validation.datasets = []  # Save time
        # config.offline_validation.q_d_tradeoff_lambda = 0.5
        config.architecture.subset_selection_strategy.gain_cutoff = -100.0
        pipeline = TrainingPipeline(config)

        query_embedding = torch.tensor([0.7071, 0.7071, 0.0000], dtype=torch.float32)
        shortlist_embeddings = torch.tensor(
            [
                [1.0000, 0.0000, 0.0000],  # 0 # partial match
                [0.7071, 0.7071, 0.0000],  # 1 # perflect quality, first pick
                [0.0000, 0.0000, 1.0000],  # 2 # completely orthogonal
                [0.7071, 0.7071, 0.0000],  # 3 # perfect quality, wrong diversity
                [-0.7071, -0.7071, 0.0000],  # 4 # anti-parallel quality
            ],
            dtype=torch.float32,
        )

        query_embedding = F.normalize(query_embedding, dim=-1)
        shortlist_embeddings = F.normalize(shortlist_embeddings, dim=-1)

        # Apply the indexes
        result, scores = pipeline.subset_selection_strategy.subset_select(
            query_embedding, shortlist_embeddings
        )

        # Expected sequence = [1, 0, 2, 3, 4]
        expected_output = [1, 3, 0, 2, 4]
        expected_scores = [
            1.0,
            0.0,
            -0.04881554841995239,
            -0.1127961277961731,
            -0.167677640914917,
        ]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.quaild_submodular_test.TestQuaildSubmodularStrategy.test_subset_select_many -v
    def test_subset_select_many(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"
        config.architecture.dense_index.type = "in_memory"
        config.offline_validation.datasets = []  # Save time
        # config.offline_validation.q_d_tradeoff_lambda = 0.5
        config.architecture.subset_selection_strategy.gain_cutoff = -100.0
        pipeline = TrainingPipeline(config)

        query_embedding = torch.tensor(
            [
                [0.7071, 0.7071, 0.0000, 0.0000],
                [0.0000, 0.7071, 0.7071, 0.0000],
            ],
            dtype=torch.float32,
        )
        shortlist_embeddings = torch.tensor(
            [
                # 0 # partial match to first
                [1.0000, 0.0000, 0.0000, 0.0000],
                # 1 # perflect quality, first pick
                [0.5774, 0.5774, 0.5774, 0.0000],
                # 2 # completely orthogonal
                [0.0000, 0.0000, 0.0000, 1.0000],
                # 3 # perfect quality, wrong diversity
                [0.5774, 0.5774, 0.5774, 0.0000],
                # 4 # anti-parallel quality
                [-0.5774, -0.5774, -0.5774, 0.0000],
                # 5 # partial match to both
                [0.0000, 1.0000, 0.0000, 0.0000],
            ],
            dtype=torch.float32,
        )

        query_embedding = F.normalize(query_embedding, dim=-1)
        shortlist_embeddings = F.normalize(shortlist_embeddings, dim=-1)

        # Apply the indexes
        result, scores = pipeline.subset_selection_strategy.subset_select(
            query_embedding, shortlist_embeddings
        )

        # Expected sequence = [1, 0, 2, 3, 4]
        expected_output = [1, 3, 5, 0, 2, 4]
        expected_scores = [
            0.9082483053207397,
            0.0,
            -0.018231570720672607,
            -0.05331003665924072,
            -0.06734126806259155,
            -0.11293572187423706,
        ]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.quaild_submodular_test.TestQuaildSubmodularStrategy.test_real_data -v
    def test_real_data(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        # config.architecture.subset_selection_strategy.gain_cutoff = 0
        config.architecture.subset_selection_strategy.k = 5
        pipeline = TrainingPipeline(config)
        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        metrics = pipeline.compute_extra_metrics(batch)

        assert metrics == {
            "precision": 0.2,
            "recall": 0.125,
            "f1_score": 0.15384615384615385,
        }, metrics

    # python -m unittest subset_selection_strategies.quaild_submodular_test.TestQuaildSubmodularStrategy.test_other_loss_types -v
    def test_other_loss_types(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"  # Save time
        config.architecture.dense_index.type = "in_memory"  # Save time
        config.offline_validation.datasets = []  # Save time
        config.training.loss.type = "mean_squared_error"
        config.architecture.subset_selection_strategy.gain_cutoff = -100.0
        pipeline = TrainingPipeline(config)

        # Create an instance of the strategy
        strategy = QuaildSubmodularStrategy(config, pipeline)

        query_embedding = torch.tensor([1, 0, 0], dtype=torch.float32)
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

        expected_output = [0, 1, 2]
        expected_scores = [1.0, 0.0, -0.3333333134651184]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.quaild_submodular_test.TestQuaildSubmodularStrategy.test_all_below_gain_cutoff -v
    def test_all_below_gain_cutoff(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.subset_selection_strategy.gain_cutoff = 1000
        config.offline_validation.datasets = []
        pipeline = TrainingPipeline(config)

        strategy = QuaildSubmodularStrategy(config, pipeline)

        query_embedding = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
        shortlist_embeddings = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.5, 0.4, 0.1],
            ],
            dtype=torch.float32,
        )

        result, scores = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = []
        expected_scores = []

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.quaild_submodular_test.TestQuaildSubmodularStrategy.test_empty_shortlist -v
    def test_empty_shortlist(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.datasets = []
        pipeline = TrainingPipeline(config)

        strategy = QuaildSubmodularStrategy(config, pipeline)

        query_embedding = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
        shortlist_embeddings = torch.tensor([], dtype=torch.float32).reshape(0, 3)

        result, scores = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = []
        expected_scores = []

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()


if __name__ == "__main__":
    unittest.main()
