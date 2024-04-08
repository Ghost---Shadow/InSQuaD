import argparse
from config import Config, RootConfig
from notifications.discord_wrapper import send_discord_notification
from offline_eval_pipeline import OfflineEvaluationPipeline


def main(config: RootConfig, dataset_name: str):
    pipeline = OfflineEvaluationPipeline(config)

    EXPERIMENT_NAME = config.wandb.name
    # TODO: Send the artifact to the same experiment as wandb

    try:
        send_discord_notification(f"Eval for {EXPERIMENT_NAME} started")

        print(f"Shortlisting")
        pipeline.shortlist(dataset_name)

        print(f"Generating one shots")
        pipeline.generate_one_shots(dataset_name)

        print(f"Running inference")
        pipeline.run_inference(dataset_name)
        send_discord_notification(f"Eval for {EXPERIMENT_NAME} finished!")
    except Exception as e:
        send_discord_notification(f"Eval for {EXPERIMENT_NAME} crashed!")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model offline based on provided config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)

    for dataset_name in config.offline_validation.datasets:
        main(config, dataset_name)
