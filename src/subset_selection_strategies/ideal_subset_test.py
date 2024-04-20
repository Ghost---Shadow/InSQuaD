import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
import torch


# python -m unittest subset_selection_strategies.ideal_subset_test.TestIdealSubsetStrategy -v
class TestIdealSubsetStrategy(unittest.TestCase):
    # python -m unittest subset_selection_strategies.ideal_subset_test.TestIdealSubsetStrategy.test_ideal_happy -v
    def test_ideal_happy(self):
        config = Config.from_file("experiments/tests/ideal_test_experiment.yaml")
        config.offline_validation.annotation_budget = 3  # number to select
        config.architecture.subset_selection_strategy.k = 4  # top k
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        embedding_matrix = torch.randn(10, 50)
        selected_indices, selected_scores = (
            pipeline.subset_selection_strategy.subset_select(embedding_matrix)
        )

        assert selected_indices == [0, 4, 7], selected_indices
        assert selected_scores == [0, 5.2, 6.3], selected_scores
