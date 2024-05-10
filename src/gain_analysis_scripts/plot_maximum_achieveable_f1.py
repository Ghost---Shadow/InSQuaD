import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_graph(plot_name, key_name):
    sns.set_theme("paper")

    plt.figure(figsize=(10, 10))
    data_list = []
    root_dir = "./artifacts"
    for dir_name, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name == f"{plot_name}.json":
                experiment_name = os.path.basename(
                    os.path.dirname(os.path.dirname(dir_name))
                )
                dir_name = Path(dir_name)
                tradeoff = float(str(dir_name.name))
                experiment_name = dir_name.parents[3].name

                with open(os.path.join(dir_name, file_name), "r") as f:
                    results = json.load(f)

                for key, f1_scores in results.items():
                    for f1 in f1_scores:
                        if key_name == "gain":
                            key = int(key)
                        else:
                            key = int(key)
                        data_list.append([f"{experiment_name}_{tradeoff}", key, f1])

    df = pd.DataFrame(data_list, columns=["Experiment", key_name, "F1"])
    df = df.sort_values(key_name, ascending=True)
    # df_grouped = df.groupby(["Experiment", key_name]).mean()
    # print(df_grouped)

    # Create a lineplot
    sns.lineplot(x=key_name, y="F1", hue="Experiment", data=df)
    # plt.ylim(top=1)

    # # Find maximum F1 score for each experiment
    # for experiment_name in df_grouped["Experiment"].unique():
    #     df_experiment = df_grouped[df_grouped["Experiment"] == experiment_name]
    #     max_f1_df = df_experiment.loc[df_experiment["F1"].idxmax()]
    #     max_f1 = max_f1_df["F1"]
    #     max_f1_score = max_f1_df[key_name]

    #     # Draw the line at the maximum F1 score for each experiment and annotate the point
    #     plt.axhline(max_f1, color="r", linestyle="dashed", linewidth=1)
    #     plt.text(
    #         max_f1_score,
    #         max_f1,
    #         f"{experiment_name} (Score: {max_f1_score:.2f}, F1: {max_f1:.2f})",
    #         color="r",
    #     )

    # # Save the plot (change the filename as per your requirement, here 'myplot.png')
    path = f"./artifacts/{plot_name}.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(path)


plot_names = ["sweep_results_gain", "sweep_results_k"]
key_names = ["gain", "k"]

for plot_name, key_name in zip(plot_names, key_names):
    plot_graph(plot_name, key_name)
