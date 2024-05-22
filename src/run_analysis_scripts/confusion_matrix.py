import json
import re
from collections import defaultdict
from dataloaders.dbpedia import DBPedia
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# Function to load data from a JSONL file and extract few-shot labels using regex for all labels
def load_and_extract_labels(filepath, regex_pattern):
    label_few_shot_counts = defaultdict(lambda: defaultdict(int))
    with open(filepath, "r") as file:
        for line in file:
            data = json.loads(line)
            main_label = data["labels"]
            # Use re.findall to capture all occurrences of the pattern
            matches = re.findall(regex_pattern, data["prompts"])
            for match in matches:
                few_shot_label = match  # match is directly the few-shot label since re.findall returns the captured groups
                label_few_shot_counts[main_label][few_shot_label] += 1
    return label_few_shot_counts


# Function to create a DataFrame from the nested dictionary and ensure alignment
def create_dataframe(label_few_shot_counts):
    df = pd.DataFrame.from_dict(label_few_shot_counts, orient="index").fillna(0)
    # Ensure the dataframe is square and indices/columns match
    all_labels = sorted(set(df.index).union(df.columns))
    df = df.reindex(index=all_labels, columns=all_labels, fill_value=0)
    df = df.div(df.sum(axis=1), axis=0)  # Normalize by row to get percentages
    return df


# Function to plot the heatmap
def plot_heatmap(df, name, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Few-Shot Labels")
    plt.ylabel("Main Labels")
    plt.title(f"Confusion Matrix of Few-Shot labels vs real label for DBPedia ({name})")
    plt.xticks(rotation=45, ha="right")
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

        # Load and extract label and few-shot label counts
        label_few_shot_counts = load_and_extract_labels(
            inference_result_path, regex_pattern
        )

        # Create a DataFrame from the counts
        df = create_dataframe(label_few_shot_counts)

        # Plotting the heatmap
        output_dir = Path("artifacts/diagrams/label_distribution")
        output_dir.mkdir(parents=True, exist_ok=True)
        name = experiments[experiment]
        lower_name = name.lower().replace("-", "_")
        plot_heatmap(df, name, output_dir / f"confusion_matrix_{lower_name}.png")
