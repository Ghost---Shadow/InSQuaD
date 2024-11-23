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
from google.cloud import storage


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

    def try_load_checkpoint(self, epoch=None, for_eval=False):
        assert self.pipeline.current_seed is not None
        try:
            self.load_checkpoint(epoch, for_eval)
        except FileNotFoundError:
            print(f"No checkpoint found at {self.checkpoint_dir} starting new run")
            return False
        return True

    def load_checkpoint(self, epoch=None, for_eval=False):
        checkpoint_files = [
            file for file in os.listdir(self.checkpoint_dir) if file.endswith(".pth")
        ]
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoints found in the directory {self.checkpoint_dir}."
            )

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
        if not for_eval:
            # Required for training but not eval
            self.pipeline.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            self.pipeline.lr_scheduler.load_state_dict(
                checkpoint.lr_scheduler_state_dict
            )
            self.pipeline.scaler.load_state_dict(checkpoint.scaler_state_dict)

            # Restoring random states
            random.setstate(checkpoint.random_state)
            np.random.set_state(checkpoint.numpy_random_state)
            torch.set_rng_state(checkpoint.torch_random_state)

            self.pipeline.current_step = checkpoint.step
            self.pipeline.current_epoch = checkpoint.epoch
            self.pipeline.current_seed = checkpoint.seed

        print(f"Checkpoint loaded from {checkpoint_path}")

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

    def try_download_from_bucket(self):
        downloaded_something = False

        client = storage.Client()
        bucket_name = "quaild-icl-bucket"
        local_dir = self.checkpoint_dir
        # Remove the seed part to match the bucket structure
        local_dir_base = "/".join(local_dir.split("/")[:-2]) + "/"
        prefix = local_dir_base.split("./")[-1]

        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            local_path = os.path.join(local_dir_base, blob.name.replace(prefix, ""))
            if not os.path.exists(local_path):  # Check if the file already exists
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Downloading: {local_path}")
                blob.download_to_filename(local_path)
                downloaded_something = True
                print(f"Downloaded: {local_path}")
            else:
                print(f"Skipped (already exists): {local_path}")

        return downloaded_something

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
