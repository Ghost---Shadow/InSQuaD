import argparse
import json
from config import Config, RootConfig
from notifications.discord_wrapper import send_discord_notification
from offline_eval_pipeline import OfflineEvaluationPipeline
from run_analysis_scripts.excelify import excelify_for_discord
import torch


def main(config: RootConfig, dataset_name: str, seed: int):
    pipeline = OfflineEvaluationPipeline(config)
    pipeline.set_seed(seed)
    pipeline.current_dataset_name = dataset_name

    EXPERIMENT_NAME = config.wandb.name
    # TODO: Send the artifact to the same experiment as wandb

    print("-" * 80)
    print(f"Starting eval for {EXPERIMENT_NAME}, Dataset: {dataset_name}, Seed: {seed}")

    if pipeline.is_done():
        print("Already done")
        pipeline.cleanup()
        return

    try:
        send_discord_notification(f"Eval for {EXPERIMENT_NAME} started")

        print(f"Shortlisting")
        pipeline.shortlist()

        print(f"Generating few shots")
        pipeline.generate_few_shots()

        print(f"Running inference")
        pipeline.run_inference()
        pipeline.analyze_inference_outputs()

        with open(pipeline.final_result_json_path) as f:
            accuracy = json.load(f)["accuracy"]

        finish_message = (
            f"Eval for {EXPERIMENT_NAME} finished with accuracy {accuracy}\n"
        )
        table_message = excelify_for_discord()

        send_discord_notification(finish_message + table_message)
    except Exception as e:
        send_discord_notification(f"Eval for {EXPERIMENT_NAME}/{dataset_name} crashed!")
        raise e
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model offline based on provided config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)

    for seed in config.offline_validation.seeds:
        for dataset_name in config.offline_validation.datasets:
            main(config, dataset_name, seed)
