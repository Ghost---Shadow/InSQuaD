import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

EXPERIMENT_NAME_LUT = {
    "votek_mpnet_gemma": "Vote-K",
    "ideal_mpnet_gemma": "IDEAL",
    "quaild_gain_fl_mpnet_gemma": "InSQuaD-FL",
    "quaild_gain_gc_mpnet_gemma": "InSQuaD-GC",
    "quaild_gain_ld_mpnet_gemma": "InSQuaD-LD",
}


def read_json_files(root_dir):
    data = []
    root_path = Path(root_dir)
    # Traverse the directory structure
    for json_file in root_path.rglob("time_result.json"):
        with open(json_file, "r") as file:
            json_data = json.load(file)
            elapsed_time = json_data["elapsed_milliseconds"]
            parts = json_file.parts
            experiment_name = parts[1]
            seed = parts[2]
            dataset_name = parts[3]

            experiment_name = "_".join(experiment_name.split("_")[:-1])

            if experiment_name not in EXPERIMENT_NAME_LUT:
                continue

            experiment_name = EXPERIMENT_NAME_LUT[experiment_name]

            data.append(
                {
                    "Experiment": experiment_name,
                    "Seed": seed,
                    "Dataset": dataset_name,
                    "Elapsed Time (ms)": elapsed_time,
                }
            )
    return data


def plot_data(data):
    # Convert list to DataFrame
    df = pd.DataFrame(data)
    # Sort data by elapsed time
    df_sorted = df.sort_values("Elapsed Time (ms)", ascending=False)

    # Enable LaTeX typesetting
    # plt.rc("text", usetex=True)
    # plt.rc("font", family="serif")

    # Set up the matplotlib figure to control size, aspect ratio
    plt.figure(figsize=(4, 2.5))
    sns.set_theme("paper")

    # Create a color palette and plot
    palette = sns.color_palette("viridis", as_cmap=False)
    ax = sns.barplot(
        data=df_sorted,
        x="Experiment",
        y="Elapsed Time (ms)",
        # hue="Dataset",
        palette=palette,
    )
    ax.set_yscale("log")  # Set the y-axis to logarithmic scale

    plt.title("Elapsed Time for Experiments (Log Scale)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    Path("./artifacts/diagrams/").mkdir(parents=True, exist_ok=True)
    plt.savefig("./artifacts/diagrams/time_comparisons.png")


if __name__ == "__main__":
    root_directory = "./artifacts"  # Adjust this to your starting directory
    data = read_json_files(root_directory)
    plot_data(data)
