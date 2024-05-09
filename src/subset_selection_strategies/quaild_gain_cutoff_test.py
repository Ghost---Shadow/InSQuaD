import torch
from config import Config
from training_pipeline import TrainingPipeline
from subset_selection_strategies.quaild_gain_cutoff import QuaildGainCutoffStrategy
import unittest
import torch.nn.functional as F


# python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy -v
class TestQuaildGainCutoffStrategy(unittest.TestCase):
    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_subset_select -v
    def test_subset_select(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"
        config.architecture.dense_index.type = "in_memory"
        config.offline_validation.datasets = []  # Save time
        config.offline_validation.q_d_tradeoff_lambda = 0.5
        config.architecture.subset_selection_strategy.gain_cutoff = 0.0
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

        # Expected sequence = [1, 0, 2, 3, 4]

        query_embedding = F.normalize(query_embedding, dim=-1)
        shortlist_embeddings = F.normalize(shortlist_embeddings, dim=-1)

        # Apply the indexes
        result, scores = pipeline.subset_selection_strategy.subset_select(
            query_embedding, shortlist_embeddings
        )

        expected_output = [1, 0, 2, 3, 4]
        expected_scores = [
            1.0000001192092896,
            0.5252508521080017,
            0.5000001192092896,
            0.4642007648944855,
            0.00018402989371679723,
        ]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_real_data -v
    def test_real_data(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.subset_selection_strategy.gain_cutoff = 0.5
        pipeline = TrainingPipeline(config)
        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        # all_text = [batch["question"][0], *batch["documents"][0]]
        # all_embeddings = pipeline.semantic_search_model.embed(all_text)
        # question_embedding = all_embeddings[0]
        # document_embeddings = all_embeddings[1:]

        # # Apply the indexes
        # result = strategy.subset_select(question_embedding, document_embeddings)

        # expected_output = [52]

        # assert result.tolist() == expected_output, result.tolist()

        metrics = pipeline.compute_extra_metrics(batch)

        assert metrics == {
            "precision": 0.16666666666666666,
            "recall": 0.75,
            "f1_score": 0.27272727272727276,
        }, metrics

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_other_loss_types -v
    def test_other_loss_types(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"  # Save time
        config.architecture.dense_index.type = "in_memory"  # Save time
        config.offline_validation.datasets = []  # Save time
        config.training.loss.type = "mean_squared_error"
        pipeline = TrainingPipeline(config)

        # Create an instance of the strategy
        strategy = QuaildGainCutoffStrategy(config, pipeline)

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

        expected_output = [0]
        expected_scores = [1.0000001192092896]

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_all_below_gain_cutoff -v
    def test_all_below_gain_cutoff(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.subset_selection_strategy.gain_cutoff = 1000
        config.offline_validation.datasets = []
        pipeline = TrainingPipeline(config)

        strategy = QuaildGainCutoffStrategy(config, pipeline)

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

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_empty_shortlist -v
    def test_empty_shortlist(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.offline_validation.datasets = []
        pipeline = TrainingPipeline(config)

        strategy = QuaildGainCutoffStrategy(config, pipeline)

        query_embedding = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
        shortlist_embeddings = torch.tensor([], dtype=torch.float32).reshape(0, 3)

        result, scores = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = []
        expected_scores = []

        assert result.tolist() == expected_output, result.tolist()
        assert scores.tolist() == expected_scores, scores.tolist()


if __name__ == "__main__":
    unittest.main()
