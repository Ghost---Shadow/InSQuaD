import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from config import Config
import dotenv
from extra_metrics.hotpotqa_with_q_f1 import ExtraMetricHotpotQaWithQF1
from tqdm import tqdm
from training_pipeline import TrainingPipeline


def main(config, seed, split, limit, q_d_tradeoff_lambda):
    dotenv.load_dotenv()

    # We want to see the full sweep, so no early stopping
    config.architecture.subset_selection_strategy.gain_cutoff = 0.0

    # Sweep to find optimal tradeoff lambda
    config.offline_validation.q_d_tradeoff_lambda = q_d_tradeoff_lambda

    # Load pipeline
    pipeline = TrainingPipeline(config)
    pipeline.set_seed(seed)

    base_path = (
        Path(pipeline.artifacts_dir) / "f1_experiments" / str(q_d_tradeoff_lambda)
    )
    base_path.mkdir(exist_ok=True, parents=True)
    if os.path.exists(base_path / "done"):
        print(f"Already done {base_path}")
        return

    pipeline.checkpoint_manager.try_load_checkpoint()
    extra_metric = ExtraMetricHotpotQaWithQF1(pipeline)

    dataloader = pipeline.wrapped_train_dataset.get_loader(split)

    accumulated_gain_results = defaultdict(list)
    accumulated_k_results = defaultdict(list)

    for batch, _ in zip(dataloader, tqdm(range(limit), desc=str(base_path))):
        assert len(batch["question"]) == 1
        result_f1, result_k = extra_metric.sweeping_pr_curve(batch)

        for score, f1_scores in result_f1.items():
            accumulated_gain_results[score].extend(f1_scores)

        for k, f1_scores in result_k.items():
            accumulated_k_results[k].extend(f1_scores)

    accumulated_gain_results = dict(accumulated_gain_results)
    accumulated_k_results = dict(accumulated_k_results)

    with open(base_path / f"sweep_results_k.json", "w") as f:
        json.dump(accumulated_k_results, f, indent=2)

    with open(base_path / f"sweep_results_gain.json", "w") as f:
        json.dump(accumulated_gain_results, f, indent=2)

    open(base_path / f"done", "w").close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep gains for maximum F1")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--split", type=str, help="train or validation", default="validation"
    )
    parser.add_argument("--limit", type=int, help="Subsample size", default=256)
    parser.add_argument("--q_d_tradeoff_lambda", type=float, help="QD-Tradeoff")
    args = parser.parse_args()
    split = args.split
    limit = args.limit
    q_d_tradeoff_lambda = args.q_d_tradeoff_lambda
    config = Config.from_file(args.config)

    for seed in config.training.seeds:
        main(config, seed, split, limit, q_d_tradeoff_lambda)
