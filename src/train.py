import argparse
from config import Config, RootConfig
from notifications.discord_wrapper import send_discord_notification
from training_pipeline import TrainingPipeline
import wandb


def main(config: RootConfig, seed: int):
    pipeline = TrainingPipeline(config)
    pipeline.checkpoint_manager.try_load_checkpoint()
    start_epoch = pipeline.current_epoch + 1
    pipeline.current_seed = seed

    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"] + f"_{seed}",
        config={
            **config,
            "checkpoint_dir": pipeline.checkpoint_dir,
            "seed": seed,
        },
        entity=config["wandb"].get("entity", None),
    )

    EXPERIMENT_NAME = config.wandb.name

    try:
        send_discord_notification(f"Experiment {EXPERIMENT_NAME} started")
        if pipeline.current_step == 0:
            print(f"Running a warmup validation")
            pipeline.run_online_validation()

        for epoch in range(start_epoch, config.training.epochs):
            print(f"Start training for epoch {epoch}, seed {seed}")
            pipeline.train_one_epoch()

            print(f"Saving checkpoint epoch {epoch}, seed {seed}")
            pipeline.checkpoint_manager.save_checkpoint()

            print(f"Starting online validation epoch {epoch}, seed {seed}")
            pipeline.run_online_validation()
        send_discord_notification(f"Experiment {EXPERIMENT_NAME} finished!")
    except Exception as e:
        send_discord_notification(f"Experiment {EXPERIMENT_NAME} crashed!")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model based on provided config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)

    for seed in config.training.seeds:
        main(config, seed)
