import torch
from config import Config
from training_pipeline import TrainingPipeline
from subset_selection_strategies.quaild_gain_cutoff import QuaildGainCutoffStrategy
import unittest


# python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy -v
class TestQuaildGainCutoffStrategy(unittest.TestCase):
    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_subset_select -v
    def test_subset_select(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.architecture.semantic_search_model.type = "noop"
        config.architecture.dense_index.type = "in_memory"
        config.offline_validation.datasets = []  # Save time
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
        result = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = [0]

        assert result.tolist() == expected_output, result.tolist()

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_real_data -v
    def test_real_data(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        pipeline = TrainingPipeline(config)
        strategy = QuaildGainCutoffStrategy(config, pipeline)
        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        all_text = [batch["question"][0], *batch["documents"][0]]
        all_embeddings = pipeline.semantic_search_model.embed(all_text)
        question_embedding = all_embeddings[0]
        document_embeddings = all_embeddings[1:]

        # Apply the indexes
        result = strategy.subset_select(question_embedding, document_embeddings)

        expected_output = [13, 52, 113]

        # print("-" * 80)
        # print(batch["question"][0])
        # print("-" * 80)
        # for idx in expected_output:
        #     print(batch["documents"][0][idx])
        # print("-" * 80)
        # for idx in batch["relevant_indexes"][0]:
        #     print(batch["documents"][0][idx])
        # print("-" * 80)

        assert result.tolist() == expected_output, result.tolist()

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_other_loss_types -v
    def test_other_loss_types(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
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
        result = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = [0]

        assert result.tolist() == expected_output, result.tolist()

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_all_below_gain_cutoff -v
    def test_all_below_gain_cutoff(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
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

        result = strategy.subset_select(query_embedding, shortlist_embeddings)

        expected_output = []

        assert result.tolist() == expected_output, result.tolist()

    # python -m unittest subset_selection_strategies.quaild_gain_cutoff_test.TestQuaildGainCutoffStrategy.test_empty_shortlist -v
    def test_empty_shortlist(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.offline_validation.datasets = []
        pipeline = TrainingPipeline(config)

        strategy = QuaildGainCutoffStrategy(config, pipeline)

        query_embedding = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
        shortlist_embeddings = torch.tensor([], dtype=torch.float32).reshape(0, 3)

        result = strategy.subset_select(query_embedding, shortlist_embeddings)

        self.assertTrue(
            result.numel() == 0, "Expected no selection with an empty shortlist."
        )


if __name__ == "__main__":
    unittest.main()
