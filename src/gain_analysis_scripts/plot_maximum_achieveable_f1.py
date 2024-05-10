import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_graph(plot_name, key_name):
    sns.set_theme("paper")
    plt.tight_layout()
    plt.figure(figsize=(10, 10))

    data_list = []
    root_dir = "./artifacts"
    for dir_name, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name == f"{plot_name}.json":
                dir_name = Path(dir_name)
                experiment_name = dir_name.parents[2].name

                with open(os.path.join(dir_name, file_name), "r") as f:
                    results = json.load(f)

                for key, f1_scores in results.items():
                    for f1 in f1_scores:
                        if key_name == "gain":
                            key = float(key)
                        else:
                            key = int(key)
                        data_list.append([f"{experiment_name}", key, f1])

    df = pd.DataFrame(data_list, columns=["Experiment", key_name, "F1"])
    df = df.sort_values(key_name, ascending=True)

    # Create a lineplot
    sns.lineplot(x=key_name, y="F1", hue="Experiment", data=df)
    # plt.ylim(top=1)

    # Calculate the average F1 score for each key value within each experiment
    df_grouped = df.groupby(["Experiment", key_name]).mean().reset_index()

    # Iterate over each experiment to find and draw the line for the maximum average F1
    for experiment in df["Experiment"].unique():
        # Filter the data for the current experiment
        df_experiment = df_grouped[df_grouped["Experiment"] == experiment]

        # Find the row with the maximum F1 score
        max_row = df_experiment.loc[df_experiment["F1"].idxmax()]
        max_f1 = max_row["F1"]
        max_key_value = max_row[key_name]

        # Draw the horizontal line at the maximum F1 score
        plt.axhline(y=max_f1, color="r", linestyle="dashed", linewidth=1)

        # Annotate with text
        plt.text(
            0,
            max_f1,
            f"(F1: {max_f1:.2f}, {key_name}: {max_key_value})",
            color="r",
            verticalalignment="bottom",
        )

    path = f"./artifacts/{plot_name}.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(path)


plot_names = ["sweep_results_gain", "sweep_results_k"]
key_names = ["gain", "k"]

for plot_name, key_name in zip(plot_names, key_names):
    plot_graph(plot_name, key_name)
