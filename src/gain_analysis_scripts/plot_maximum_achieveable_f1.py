import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def format_experiment_name(experiment_name):
    EXPERIMENT_NAME_LUT = {
        "quaild_gain_fl_mpnet_stablelm": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm": "QuailD-GC",
        "quaild_nt_fl_mpnet_stablelm": "QuailD-FL (NT)",
        "quaild_nt_gc_mpnet_stablelm": "QuailD-GC (NT)",
    }
    experiment_name = "_".join(experiment_name.split("_")[:-1])
    return EXPERIMENT_NAME_LUT.get(experiment_name, experiment_name)


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
                            # key = float(f"{key:.2f}")  # larger buckets
                        else:
                            key = int(key)
                        formatted_experiment_name = format_experiment_name(
                            f"{experiment_name}"
                        )
                        data_list.append([formatted_experiment_name, key, f1])

    df = pd.DataFrame(data_list, columns=["Experiment", key_name, "F1"])
    df = df.sort_values(key_name, ascending=True)
    df = df[df[key_name] > -0.002]
    # df.to_csv(f"{key_name}.csv")

    # Create a lineplot
    sns.lineplot(x=key_name, y="F1", hue="Experiment", data=df)
    plt.ylim(top=0.35)

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
            -0.00175,
            max_f1,
            f"(F1: {max_f1:.2f}, {key_name}: {max_key_value})",
            color="r",
            verticalalignment="bottom",
        )

    Path("./artifacts/diagrams/").mkdir(parents=True, exist_ok=True)
    path = f"./artifacts/diagrams/{plot_name}.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(path)


if __name__ == "__main__":
    plot_names = ["sweep_results_gain", "sweep_results_k"]
    key_names = ["gain", "k"]

    for plot_name, key_name in zip(plot_names, key_names):
        plot_graph(plot_name, key_name)
