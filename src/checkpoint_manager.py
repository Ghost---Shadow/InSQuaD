import os
from pathlib import Path
import random
import re
import numpy as np
import torch
from pydantic import BaseModel
from typing import Any
from train_utils import generate_md5_hash
import yaml


class Checkpoint(BaseModel):
    epoch: Any
    step: Any
    seed: Any
    semantic_search_model_state_dict: Any
    optimizer_state_dict: Any
    config: Any
    random_state: Any
    numpy_random_state: Any
    torch_random_state: Any
    lr_scheduler_state_dict: Any
    scaler_state_dict: Any


class CheckpointManager:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = self.pipeline.config

    def get_latest_checkpoint(self, checkpoint_files):
        checkpoint_files = sorted(
            checkpoint_files,
            key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)),
        )
        latest_checkpoint = checkpoint_files[-1]
        return os.path.join(self.checkpoint_dir, latest_checkpoint)

    def try_load_checkpoint(self, epoch=None):
        assert self.pipeline.current_seed is not None
        try:
            self.load_checkpoint(epoch)
        except FileNotFoundError:
            print(f"No checkpoint found at {self.checkpoint_dir} starting new run")
            return False
        return True

    def load_checkpoint(self, epoch=None):
        checkpoint_files = [
            file for file in os.listdir(self.checkpoint_dir) if file.endswith(".pth")
        ]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoints found in the directory.")

        if epoch is None:
            checkpoint_path = self.get_latest_checkpoint(checkpoint_files)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"No checkpoint found for epoch {epoch}.")

        # Load checkpoint
        checkpoint_dict = torch.load(checkpoint_path)
        checkpoint = Checkpoint(**checkpoint_dict)

        # Load State dicts
        self.pipeline.semantic_search_model.model.load_state_dict(
            checkpoint.semantic_search_model_state_dict
        )
        self.pipeline.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.pipeline.lr_scheduler.load_state_dict(checkpoint.lr_scheduler_state_dict)
        self.pipeline.scaler.load_state_dict(checkpoint.scaler_state_dict)

        # Restoring random states
        random.setstate(checkpoint.random_state)
        np.random.set_state(checkpoint.numpy_random_state)
        torch.set_rng_state(checkpoint.torch_random_state)

        print(f"Checkpoint loaded from {checkpoint_path}")
        self.pipeline.current_step = checkpoint.step
        self.pipeline.current_epoch = checkpoint.epoch
        self.pipeline.current_seed = checkpoint.seed

    @property
    def checkpoint_dir(self):
        seed = self.pipeline.current_seed
        # Reuse trained models with same training config
        train_config = self.config.training.model_dump()
        del train_config["extra_metrics"]
        del train_config["seeds"]
        config_name_with_hash = generate_md5_hash(train_config)
        checkpoint_dir = f"./checkpoints/{config_name_with_hash}/seed_{seed}/"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        config_path = Path(checkpoint_dir) / "config.yaml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.safe_dump(train_config, f)
        return checkpoint_dir

    def save_checkpoint(self):
        epoch = self.pipeline.current_epoch
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        checkpoint = Checkpoint(
            epoch=epoch,
            step=self.pipeline.current_step,
            seed=self.pipeline.current_seed,
            semantic_search_model_state_dict=self.pipeline.semantic_search_model.model.state_dict(),
            optimizer_state_dict=self.pipeline.optimizer.state_dict(),
            config=self.config,
            random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            torch_random_state=torch.get_rng_state(),
            lr_scheduler_state_dict=self.pipeline.lr_scheduler.state_dict(),
            scaler_state_dict=self.pipeline.scaler.state_dict(),
        )

        torch.save(checkpoint.model_dump(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
