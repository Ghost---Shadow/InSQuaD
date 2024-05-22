import json
from dataloaders.dbpedia import DBPedia
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


# Function to load data from a JSON file
def load_data(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


# Function to extract labels from data
def extract_labels(data):
    return [item["labels"] for item in data]


# Function to calculate percentages
def calculate_percentage(counts, total):
    return {label: (count / total) * 100 for label, count in counts.items()}


# Function to plot combined label distribution as percentages
def plot_combined_label_distribution(dataframe, output_path):
    plt.figure(figsize=(12, 8))
    sns.set_theme("paper")  # Set the seaborn theme to 'paper'
    sns.barplot(
        x="Label", y="Percentage", hue="Stage", data=dataframe, palette="viridis"
    )
    plt.xlabel("Labels")
    plt.ylabel("Percentage (%)")
    plt.title("Combined Label Distribution as Percentages across Stages")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    experiments = {
        "quaild_gain_fl_mpnet_gemma_33940": "Quaild-FL",
        "quaild_gain_gc_mpnet_gemma_e7ae2": "Quaild-GC",
        "quaild_gain_ld_mpnet_gemma_1358e": "Quaild-LD",
    }

    combined_data = []

    for experiment in experiments:
        base_path = Path(f"artifacts/{experiment}/seed_42/{DBPedia.NAME}")

        # Load and extract labels for long list
        long_list_path = base_path / "longlist.json"
        long_list_data = load_data(long_list_path)
        long_list_labels = extract_labels(long_list_data)
        long_list_label_counts = Counter(long_list_labels)
        long_list_total = sum(long_list_label_counts.values())
        long_list_percentages = calculate_percentage(
            long_list_label_counts, long_list_total
        )

        # Load and extract labels for shortlist
        shortlist_path = base_path / "shortlist.json"
        shortlist_data = load_data(shortlist_path)
        shortlist_labels = extract_labels(shortlist_data)
        shortlist_label_counts = Counter(shortlist_labels)
        shortlist_total = sum(shortlist_label_counts.values())
        shortlist_percentages = calculate_percentage(
            shortlist_label_counts, shortlist_total
        )

        # Load and extract labels for Eval List (inference results)
        inference_result_path = base_path / "inference_result.jsonl"
        eval_list_data = [json.loads(line) for line in open(inference_result_path, "r")]
        eval_list_labels = extract_labels(eval_list_data)
        eval_list_label_counts = Counter(eval_list_labels)
        eval_list_total = sum(eval_list_label_counts.values())
        eval_list_percentages = calculate_percentage(
            eval_list_label_counts, eval_list_total
        )

        # Aggregate the data for the combined plot
        for label, percentage in long_list_percentages.items():
            combined_data.append(
                {
                    "Label": label,
                    "Percentage": percentage,
                    "Stage": "Long List",
                    "experiment": experiment,
                }
            )
        for label, percentage in shortlist_percentages.items():
            combined_data.append(
                {
                    "Label": label,
                    "Percentage": percentage,
                    "Stage": "Short List",
                    "experiment": experiment,
                }
            )
        for label, percentage in eval_list_percentages.items():
            combined_data.append(
                {
                    "Label": label,
                    "Percentage": percentage,
                    "Stage": "Eval List",
                    "experiment": experiment,
                }
            )

    # Convert the combined data to a DataFrame
    combined_df = pd.DataFrame(combined_data)

    # Plotting combined label distributions
    output_dir = Path("artifacts/diagrams/label_distribution")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_combined_label_distribution(
        combined_df, output_dir / "label_distribution_by_stages.png"
    )
