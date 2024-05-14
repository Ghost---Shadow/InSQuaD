import argparse
from datetime import datetime
import json
from pathlib import Path
import traceback
from config import Config
from notifications.discord_wrapper import send_discord_notification
from offline_eval_pipeline import OfflineEvaluationPipeline
from run_analysis_scripts.excelify import excelify_for_discord
from training_strategies.no_operation import NoOperation


def main(pipeline: OfflineEvaluationPipeline, dataset_name: str, seed: int):
    pipeline.set_seed(seed)
    pipeline.current_dataset_name = dataset_name

    EXPERIMENT_NAME = config.wandb.name
    # TODO: Send the artifact to the same experiment as wandb

    print("-" * 80)
    print(f"Starting eval for {EXPERIMENT_NAME}, Dataset: {dataset_name}, Seed: {seed}")

    if pipeline.is_done():
        print("Already done")
        return

    if config.training.type != NoOperation.NAME:
        # Checkpoint must exist or it should crash and exit
        pipeline.checkpoint_manager.load_checkpoint(for_eval=True)

    try:
        send_discord_notification(f"Eval for {EXPERIMENT_NAME} started")

        print(f"Shortlisting")
        with pipeline.timer():
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
        # Format the traceback
        error_trace = traceback.format_exc()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Write the traceback to a file
        Path("./artifacts/crashes/").mkdir(exist_ok=True, parents=True)
        file_path = (
            f"./artifacts/crashes/{EXPERIMENT_NAME}_{dataset_name}_{timestamp}.txt"
        )
        with open(file_path, "w") as file:
            file.write(error_trace)

        # Send a notification to Discord
        send_discord_notification(
            f"Eval for {EXPERIMENT_NAME}/{dataset_name} crashed!\n\n{error_trace}"
        )
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model offline based on provided config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)
    pipeline = OfflineEvaluationPipeline(config)

    for seed in config.offline_validation.seeds:
        for dataset_name in config.offline_validation.datasets:
            main(pipeline, dataset_name, seed)
