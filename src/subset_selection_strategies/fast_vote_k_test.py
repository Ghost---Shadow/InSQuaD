import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
import torch


# python -m unittest subset_selection_strategies.fast_vote_k_test.TestFastVoteK -v
class TestFastVoteK(unittest.TestCase):
    # python -m unittest subset_selection_strategies.fast_vote_k_test.TestFastVoteK.test_fast_vote_k_happy -v
    def test_fast_vote_k_happy(self):
        config = Config.from_file("experiments/tests/fastvotek_test_experiment.yaml")
        config.offline_validation.annotation_budget = 3  # number to select
        config.architecture.subset_selection_strategy.k = 4  # top k
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        embedding_matrix = torch.randn(10, 50)
        selected_indices, selected_scores = (
            pipeline.subset_selection_strategy.subset_select(embedding_matrix)
        )

        assert selected_indices == [0, 2, 7], selected_indices
        assert selected_scores == [6, 2.2, 1.2000000000000002], selected_indices
