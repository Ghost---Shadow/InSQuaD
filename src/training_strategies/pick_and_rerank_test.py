import unittest
from config import Config
import torch
from train_utils import set_seed
from training_pipeline import TrainingPipeline
from training_strategies.pick_and_rerank import PickAndRerankStrategy


# python -m unittest training_strategies.pick_and_rerank_test.TestPickAndRankStrategy -v
class TestPickAndRankStrategy(unittest.TestCase):
    # python -m unittest training_strategies.pick_and_rerank_test.TestPickAndRankStrategy.test_before_each_epoch -v
    def test_before_each_epoch(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        pipeline = TrainingPipeline(config)

        training_strategy = PickAndRerankStrategy(config, pipeline)

        # Should not crash
        training_strategy.before_each_epoch()

        assert pipeline.dense_index.index.ntotal == 3, pipeline.dense_index.index.ntotal

        # Do it again
        training_strategy.before_each_epoch()

        assert pipeline.dense_index.index.ntotal == 3, pipeline.dense_index.index.ntotal

    # python -m unittest training_strategies.pick_and_rerank_test.TestPickAndRankStrategy.test_train_step -v
    def test_train_step(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        pipeline = TrainingPipeline(config)
        training_strategy = PickAndRerankStrategy(config, pipeline)
        training_strategy.before_each_epoch()

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        # Should not crash
        loss = training_strategy.train_step(batch)

        # print(loss)

        # Also should not crash
        loss.backward()

    # python -m unittest training_strategies.pick_and_rerank_test.TestPickAndRankStrategy.test_train_step_amp -v
    def test_train_step_amp(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        pipeline = TrainingPipeline(config)
        training_strategy = PickAndRerankStrategy(config, pipeline)
        training_strategy.before_each_epoch()

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        # Should not crash
        with torch.cuda.amp.autocast():
            loss = training_strategy.train_step(batch)

        print(loss)

        # Also should not crash
        self.scaler.scale(loss).backward()
        # self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(
        #     self.semantic_search_model.get_all_trainable_parameters(), 1.0
        # )
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        # self.optimizer.zero_grad()

    # python -m unittest training_strategies.pick_and_rerank_test.TestPickAndRankStrategy.test_overfit -v
    def test_overfit(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        pipeline = TrainingPipeline(config)
        training_strategy = PickAndRerankStrategy(config, pipeline)
        training_strategy.before_each_epoch()

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        optimizer = pipeline.optimizer

        # Should not crash
        for _ in range(100):
            optimizer.zero_grad()
            loss = training_strategy.train_step(batch)
            print(loss)
            loss.backward()
            optimizer.step()
