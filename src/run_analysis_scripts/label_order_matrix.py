import json
import re
from collections import defaultdict
from dataloaders.dbpedia import DBPedia
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# Updated function to handle positions of few-shot labels that match the main label
def load_and_extract_label_positions(filepath, regex_pattern):
    label_position_counts = defaultdict(lambda: defaultdict(int))
    with open(filepath, "r") as file:
        for line in file:
            data = json.loads(line)
            main_label = data["labels"]
            # Extract all few-shot label occurrences
            matches = re.findall(regex_pattern, data["prompts"])
            for i, match in enumerate(matches):
                if match == main_label:
                    label_position_counts[main_label][i + 1] += 1
            #     print(match)
            # print("-----")
    return label_position_counts


# Function to create a DataFrame from the nested dictionary
def create_dataframe(label_position_counts):
    df = pd.DataFrame.from_dict(label_position_counts, orient="index").fillna(0)
    max_positions = max(
        (max(d.keys()) for d in label_position_counts.values()), default=0
    )
    all_positions = range(1, max_positions + 1)
    df = df.reindex(columns=all_positions, fill_value=0)
    df = df.div(df.sum(axis=1), axis=0)  # Normalize by row to get percentages
    return df


# Function to plot the heatmap
def plot_heatmap(df, name, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Position of Few-Shot Label")
    plt.ylabel("Main Labels")
    plt.title(f"Heatmap of Few-Shot Label Positions ({name})")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    experiments = {
        "quaild_gain_fl_mpnet_gemma_33940": "Quaild-FL",
        "quaild_gain_gc_mpnet_gemma_e7ae2": "Quaild-GC",
        "quaild_gain_ld_mpnet_gemma_1358e": "Quaild-LD",
    }

    for experiment in experiments:
        base_path = Path(f"artifacts/{experiment}/seed_42/{DBPedia.NAME}")
        inference_result_path = base_path / "inference_result.jsonl"
        regex_pattern = r"Topic:\nA: (.*)\n"

        # Load and extract label and their matching positions
        label_position_counts = load_and_extract_label_positions(
            inference_result_path, regex_pattern
        )

        # Create a DataFrame from the counts
        df = create_dataframe(label_position_counts)

        # Plotting the heatmap
        output_dir = Path("artifacts/diagrams/label_distribution")
        name = experiments[experiment]
        lower_name = name.lower().replace("-", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_heatmap(df, name, output_dir / f"label_position_heatmap_{lower_name}.png")
