import unittest
from config import Config
import torch
from train_utils import set_seed
from training_pipeline import TrainingPipeline
from training_strategies.quaild import QuaildStrategy
from torch.cuda.amp import autocast, GradScaler


# python -m unittest training_strategies.quaild_test.TestQuaildStrategy -v
class TestQuaildStrategy(unittest.TestCase):
    # python -m unittest training_strategies.quaild_test.TestQuaildStrategy.test_train_step -v
    def test_train_step(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        pipeline = TrainingPipeline(config)
        training_strategy = QuaildStrategy(config, pipeline)
        training_strategy.before_each_epoch()

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        # Should not crash
        loss = training_strategy.train_step(batch)

        # print(loss)

        # Also should not crash
        loss.backward()

    # python -m unittest training_strategies.quaild_test.TestQuaildStrategy.test_train_step_amp -v
    def test_train_step_amp(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        pipeline = TrainingPipeline(config)
        training_strategy = QuaildStrategy(config, pipeline)
        training_strategy.before_each_epoch()

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        scaler = GradScaler()

        # Should not crash
        with autocast():
            loss = training_strategy.train_step(batch)

        print(loss)

        # Also should not crash
        scaler.scale(loss).backward()
        # self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(
        #     self.semantic_search_model.get_all_trainable_parameters(), 1.0
        # )
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        # self.optimizer.zero_grad()

    # python -m unittest training_strategies.quaild_test.TestQuaildStrategy.test_overfit -v
    def test_overfit(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.training.loss.type = "mean_squared_error"
        pipeline = TrainingPipeline(config)
        training_strategy = QuaildStrategy(config, pipeline)
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

    # python -m unittest training_strategies.quaild_test.TestQuaildStrategy.test_amp_overfit -v
    def test_amp_overfit(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.training.loss.type = "mean_squared_error"
        pipeline = TrainingPipeline(config)
        training_strategy = QuaildStrategy(config, pipeline)
        training_strategy.before_each_epoch()

        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        optimizer = pipeline.optimizer
        # Initialize AMP scaler for managing scaled gradients
        scaler = torch.cuda.amp.GradScaler()

        # Should not crash
        for _ in range(100):
            optimizer.zero_grad()

            # Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                loss = training_strategy.train_step(batch)
                print(loss)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # Unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()
