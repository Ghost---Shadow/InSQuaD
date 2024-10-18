import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def format_experiment_name(experiment_name):
    EXPERIMENT_NAME_LUT = {
        "quaild_gain_fl_mpnet_gemma": "InSQuaD-FL",
        "quaild_gain_gc_mpnet_gemma": "InSQuaD-GC",
        "quaild_gain_ld_mpnet_gemma": "InSQuaD-LD",
        "quaild_nt_fl_mpnet_gemma": "InSQuaD-FL (NT)",
        "quaild_nt_gc_mpnet_gemma": "InSQuaD-GC (NT)",
        "quaild_nt_ld_mpnet_gemma": "InSQuaD-LD (NT)",
        # Lambda
        # "quaild_gain_fl_mpnet_gemma_lambda_0": "InSQuaD-FL 0",
        # "quaild_gain_gc_mpnet_gemma_lambda_0": "InSQuaD-GC 0",
        # "quaild_gain_ld_mpnet_gemma_lambda_0": "InSQuaD-LD 0",
        # "quaild_gain_fl_mpnet_gemma_lambda_025": "InSQuaD-FL 0.25",
        # "quaild_gain_gc_mpnet_gemma_lambda_025": "InSQuaD-GC 0.25",
        # "quaild_gain_ld_mpnet_gemma_lambda_025": "InSQuaD-LD 0.25",
        # "quaild_gain_fl_mpnet_gemma_lambda_1": "InSQuaD-FL 1",
        # "quaild_gain_gc_mpnet_gemma_lambda_1": "InSQuaD-GC 1",
        # "quaild_gain_ld_mpnet_gemma_lambda_1": "InSQuaD-LD 1",
    }
    experiment_name = "_".join(experiment_name.split("_")[:-1])
    found = experiment_name in EXPERIMENT_NAME_LUT
    return EXPERIMENT_NAME_LUT.get(experiment_name, experiment_name), found


def populate_dataframe(plot_name, key_name):
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
                        formatted_experiment_name, found = format_experiment_name(
                            f"{experiment_name}"
                        )
                        data_list.append([formatted_experiment_name, key, f1, found])
    df = pd.DataFrame(data_list, columns=["Experiment", key_name, "F1", "should_plot"])
    df = df[df["should_plot"]]
    df = df.sort_values(key_name, ascending=True)
    if key_name == "gain":
        df = df[df[key_name] > -0.003]
        df = df[df[key_name] < 0.001]
    # df.to_csv(f"{key_name}.csv")
    return df


def plot_ahlines(key_name, df, df_grouped, ax=None):
    if ax is None:
        ax = plt.gca()  # Get current axis if none is provided

    for experiment in df["Experiment"].unique():
        # Filter the data for the current experiment
        df_experiment = df_grouped[df_grouped["Experiment"] == experiment]

        # Find the row with the maximum F1 score
        max_row = df_experiment.loc[df_experiment["F1"].idxmax()]
        max_f1 = max_row["F1"]
        max_key_value = max_row[key_name]

        print(experiment, max_key_value, max_f1)

        # Draw the horizontal line at the maximum F1 score on the specified axis
        ax.axhline(y=max_f1, color="r", linestyle="dashed", linewidth=1)

        # Annotate with text on the specified axis
        ax.text(
            -0.00175,
            max_f1,
            f"(F1: {max_f1:.2f}, {key_name}: {max_key_value})",
            color="r",
            verticalalignment="bottom",
            # transform=ax.transAxes,  # Use axis transform for positioning
        )


def plot_graph(plot_name, key_name):
    # Enable LaTeX typesetting
    # plt.rc("text", usetex=True)
    # plt.rc("font", family="serif")
    sns.set_theme("paper")
    plt.tight_layout()
    plt.figure(figsize=(6, 6))

    df = populate_dataframe(plot_name, key_name)

    # Create a lineplot
    sns.lineplot(x=key_name, y="F1", hue="Experiment", data=df)
    plt.ylim(top=0.35)

    # Calculate the average F1 score for each key value within each experiment
    df_grouped = df.groupby(["Experiment", key_name]).mean().reset_index()

    # Iterate over each experiment to find and draw the line for the maximum average F1
    plot_ahlines(key_name, df, df_grouped)

    Path("./artifacts/diagrams/").mkdir(parents=True, exist_ok=True)
    path = f"./artifacts/diagrams/{plot_name}.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(path)


def plot_graph_stacked(plot_names, key_names, titles):
    sns.set_theme("paper")
    plt.tight_layout()
    # plt.figure(figsize=(6, 6))

    # Determine the number of plots
    num_plots = len(plot_names)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)

    for i, (plot_name, key_name, title_name) in enumerate(
        zip(plot_names, key_names, titles)
    ):
        df = populate_dataframe(plot_name, key_name)

        # Create a lineplot on subplot i
        sns.lineplot(ax=axes[i], x=key_name, y="F1", hue="Experiment", data=df)
        axes[i].set_ylim(top=0.35)
        axes[i].set_title(title_name)

        # Calculate the average F1 score for each key value within each experiment
        df_grouped = df.groupby(["Experiment", key_name]).mean().reset_index()

        # Iterate over each experiment to find and draw the line for the maximum average F1
        plot_ahlines(key_name, df, df_grouped, ax=axes[i])

    plt.tight_layout()
    Path("./artifacts/diagrams/").mkdir(parents=True, exist_ok=True)
    path = "./artifacts/diagrams/sweep_results_stacked.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(path)


if __name__ == "__main__":
    plot_names = ["sweep_results_gain", "sweep_results_k"]
    key_names = ["gain", "k"]
    titles = ["Sweep by Gain", "Sweep by k"]

    for plot_name, key_name in zip(plot_names, key_names):
        plot_graph(plot_name, key_name)

    plot_graph_stacked(plot_names, key_names, titles)
