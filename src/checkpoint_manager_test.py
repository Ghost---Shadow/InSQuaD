from pathlib import Path
import unittest
from checkpoint_manager import CheckpointManager
from config import Config
import torch
import random
from tqdm import tqdm
from train_utils import rmrf_if_possible
from training_pipeline import TrainingPipeline


# python -m unittest checkpoint_manager_test.TestCheckpointManager -v
class TestCheckpointManager(unittest.TestCase):
    # python -m unittest checkpoint_manager_test.TestCheckpointManager.test_checkpoint_save_load -v
    def test_checkpoint_save_load(self):
        config_path = "experiments/quaild_test_experiment.yaml"
        config = Config.from_file(config_path)
        pipeline = TrainingPipeline(config)
        pipeline.current_seed = 42

        # Wipe checkpoint dir to test clean load
        checkpoint_dir = pipeline.checkpoint_manager.checkpoint_dir
        rmrf_if_possible(checkpoint_dir)

        # There should not be any checkpoint, should return false
        assert not pipeline.checkpoint_manager.try_load_checkpoint()

        # Save checkpoint
        pipeline.checkpoint_manager.save_checkpoint()

        # Load it
        new_pipeline = TrainingPipeline(config)
        new_pipeline.current_seed = 42
        assert new_pipeline.checkpoint_manager.try_load_checkpoint()
        self.assertEqual(new_pipeline.current_epoch, 0)

        # Load it again (idempotency)
        new_pipeline.current_epoch = 5  # Peturb
        new_pipeline = TrainingPipeline(config)
        new_pipeline.current_seed = 42
        assert new_pipeline.checkpoint_manager.try_load_checkpoint()
        self.assertEqual(new_pipeline.current_epoch, 0)

        # Cleanup
        rmrf_if_possible(checkpoint_dir)

    # python -m unittest checkpoint_manager_test.TestCheckpointManager.test_get_latest_checkpoint -v
    def test_get_latest_checkpoint(self):
        config_path = "experiments/quaild_test_experiment.yaml"
        config = Config.from_file(config_path)
        pipeline = TrainingPipeline(config)
        pipeline.current_seed = 42

        checkpoint_files = ["epoch_1.pth", "epoch_11.pth", "epoch_2.pth"]
        checkpoint_manager = pipeline.checkpoint_manager

        latest_checkpoint = checkpoint_manager.get_latest_checkpoint(checkpoint_files)
        expected_checkpoint = Path("./epoch_11.pth")
        latest_checkpoint = Path(latest_checkpoint)
        self.assertEqual(latest_checkpoint.stem, expected_checkpoint.stem)


# python -m unittest checkpoint_manager_test.TestCheckpointManagerTraining -v
class TestCheckpointManagerTraining(unittest.TestCase):
    def test_training_checkpoint(self):
        config_path = "experiments/quaild_test_experiment.yaml"
        config = Config.from_file(config_path)
        epochs = 20
        interrupt_epoch = 10
        config.training.epochs = epochs
        pipeline = TrainingPipeline(config)
        pipeline.current_seed = 42

        # Wipe checkpoint dir to test clean load
        checkpoint_dir = pipeline.checkpoint_manager.checkpoint_dir
        rmrf_if_possible(checkpoint_dir)

        # Data
        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        initial_losses = []
        reloaded_losses = []

        # Train for 20 steps and drop checkpoint at step 10
        for epoch in tqdm(range(epochs)):
            pipeline.current_epoch = epoch

            pipeline.optimizer.zero_grad()
            loss = pipeline.training_strategy.train_step(batch)
            loss.backward()
            pipeline.optimizer.step()

            if epoch == interrupt_epoch - 1:
                # Save checkpoint at step 10
                pipeline.checkpoint_manager.save_checkpoint()

            if epoch > interrupt_epoch - 1:
                initial_losses.append(loss.item())

        # New pipeline
        del pipeline.semantic_search_model
        del pipeline
        new_pipeline = TrainingPipeline(config)
        new_pipeline.current_seed = 42

        # Reload checkpoint at step 10 and retrain
        new_pipeline.checkpoint_manager.try_load_checkpoint()
        assert (
            new_pipeline.current_epoch == interrupt_epoch - 1
        ), new_pipeline.current_epoch

        for epoch in tqdm(range(interrupt_epoch, epochs)):
            new_pipeline.current_epoch = epoch
            new_pipeline.optimizer.zero_grad()
            loss = new_pipeline.training_strategy.train_step(batch)
            loss.backward()
            new_pipeline.optimizer.step()

            reloaded_losses.append(loss.item())

        # Compare the loss values
        for initial_loss, reloaded_loss in zip(initial_losses, reloaded_losses):
            print(initial_loss, reloaded_loss)
            self.assertAlmostEqual(initial_loss, reloaded_loss, places=5)


# python -m unittest checkpoint_manager_test.TestCheckpointManagerTrainingAmp -v
class TestCheckpointManagerTrainingAmp(unittest.TestCase):
    def test_training_checkpoint(self):
        config_path = "experiments/quaild_test_experiment.yaml"
        config = Config.from_file(config_path)
        epochs = 20
        interrupt_epoch = 10
        config.training.epochs = epochs
        pipeline = TrainingPipeline(config)
        pipeline.current_seed = 42

        # Wipe checkpoint dir to test clean load
        checkpoint_dir = pipeline.checkpoint_manager.checkpoint_dir
        rmrf_if_possible(checkpoint_dir)

        # Data
        train_loader = pipeline.wrapped_train_dataset.get_loader("train")
        batch = next(iter(train_loader))

        initial_losses = []
        reloaded_losses = []

        # Train for 20 steps and drop checkpoint at step 10
        for epoch in tqdm(range(epochs)):
            pipeline.current_epoch = epoch

            pipeline.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = pipeline.training_strategy.train_step(batch)
            pipeline.scaler.scale(loss).backward()
            pipeline.scaler.step(pipeline.optimizer)
            pipeline.scaler.update()

            if epoch == interrupt_epoch - 1:
                # Save checkpoint at step 10
                pipeline.checkpoint_manager.save_checkpoint()

            if epoch > interrupt_epoch - 1:
                initial_losses.append(loss.item())

        # New pipeline
        del pipeline.semantic_search_model
        del pipeline
        new_pipeline = TrainingPipeline(config)
        new_pipeline.current_seed = 42

        # Reload checkpoint at step 10 and retrain
        new_pipeline.checkpoint_manager.try_load_checkpoint()
        assert (
            new_pipeline.current_epoch == interrupt_epoch - 1
        ), new_pipeline.current_epoch

        for epoch in tqdm(range(interrupt_epoch, epochs)):
            new_pipeline.current_epoch = epoch

            new_pipeline.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = new_pipeline.training_strategy.train_step(batch)
            new_pipeline.scaler.scale(loss).backward()
            new_pipeline.scaler.step(new_pipeline.optimizer)
            new_pipeline.scaler.update()

            reloaded_losses.append(loss.item())

        # Compare the loss values
        for initial_loss, reloaded_loss in zip(initial_losses, reloaded_losses):
            print(initial_loss, reloaded_loss)
            self.assertAlmostEqual(initial_loss, reloaded_loss, places=5)


if __name__ == "__main__":
    unittest.main()
