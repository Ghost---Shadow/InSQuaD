import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm


# python -m unittest shortlist_strategies.base_test.TestBaseStrategy -v
class TestBaseStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.base_test.TestBaseStrategy.test_subsample_dataset -v
    def test_subsample_dataset(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        wrapped_dataset = pipeline.offline_dataset_lut["mrpc"]

        shortlist_strategy = BaseStrategy(config, pipeline)

        # pipeline.set_seed(42) # Should set its own seed
        subsampled_train_idxs_42_1, _iterator = shortlist_strategy.subsample_dataset(
            wrapped_dataset, "train", 3000
        )
        # pipeline.set_seed(42) # Should set its own seed
        subsampled_train_idxs_42_2, _iterator = shortlist_strategy.subsample_dataset(
            wrapped_dataset, "train", 3000
        )

        # Seed stability for 42
        assert subsampled_train_idxs_42_1 == subsampled_train_idxs_42_2, (
            subsampled_train_idxs_42_1,
            subsampled_train_idxs_42_2,
        )

        pipeline.set_seed(43)
        # pipeline.set_seed(43)  # Should set its own seed
        subsampled_train_idxs_43_1, _iterator = shortlist_strategy.subsample_dataset(
            wrapped_dataset, "train", 3000
        )
        # pipeline.set_seed(43)  # Should set its own seed
        subsampled_train_idxs_43_2, _iterator = shortlist_strategy.subsample_dataset(
            wrapped_dataset, "train", 3000
        )
        #
        # Seed stability for 42
        assert subsampled_train_idxs_43_1 == subsampled_train_idxs_43_2, (
            subsampled_train_idxs_43_1,
            subsampled_train_idxs_43_2,
        )

        # Should change if seed changes
        assert subsampled_train_idxs_42_1 != subsampled_train_idxs_43_1, (
            subsampled_train_idxs_42_1,
            subsampled_train_idxs_43_1,
        )

        for row in tqdm(_iterator, total=len(subsampled_train_idxs_42_2)):
            assert "prompts" in row, row
            assert "labels" in row, row

    # python -m unittest shortlist_strategies.base_test.TestBaseStrategy.test_subsample_dataset_for_train -v
    def test_subsample_dataset_for_train(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        longlist_rows = pipeline.shortlist_strategy.subsample_dataset_for_train()
        assert len(longlist_rows) == len(
            pipeline.shortlist_strategy.subsampled_train_idxs
        )

        for row in tqdm(longlist_rows):
            assert "prompts" in row, row
            assert "labels" in row, row

    # python -m unittest shortlist_strategies.base_test.TestBaseStrategy.test_subsample_dataset_for_eval -v
    def test_subsample_dataset_for_eval(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        eval_list_rows = pipeline.shortlist_strategy.subsample_dataset_for_eval()

        assert len(eval_list_rows) == len(
            pipeline.shortlist_strategy.subsampled_eval_idxs
        )

        for row in tqdm(eval_list_rows):
            assert "prompts" in row, row
            assert "labels" in row, row
