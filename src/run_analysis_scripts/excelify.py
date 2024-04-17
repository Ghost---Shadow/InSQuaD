import os
import json
import pandas as pd


def excelify():
    # Base directory containing all the data
    BASE_DIR = "artifacts"

    # Prepare to collect all data in a list
    data = []

    # Walk through the directory structure
    for method in os.listdir(BASE_DIR):
        method_path = os.path.join(BASE_DIR, method)
        if os.path.isdir(method_path):
            for seed in os.listdir(method_path):
                seed_path = os.path.join(method_path, seed)
                if os.path.isdir(seed_path):
                    for dataset in os.listdir(seed_path):
                        result_file = os.path.join(
                            seed_path, dataset, "final_result.json"
                        )
                        if os.path.exists(result_file):
                            with open(result_file, "r") as file:
                                result_data = json.load(file)
                                if "accuracy" in result_data:
                                    # Append the data with method, seed, dataset, and accuracy
                                    data.append(
                                        {
                                            "method": method,
                                            "seed": seed,
                                            "dataset": dataset,
                                            "accuracy": result_data["accuracy"],
                                        }
                                    )

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Check if DataFrame is not empty
    if not df.empty:
        # Group by method and dataset, and average the accuracy across seeds
        df_grouped = (
            df.groupby(["method", "dataset"]).agg({"accuracy": "mean"}).reset_index()
        )

        # Pivot the DataFrame
        pivot_df = df_grouped.pivot(
            index="method", columns="dataset", values="accuracy"
        )

        # Print the pivoted DataFrame
        return pivot_df
    else:
        return None


def excelify_for_discord():
    df = excelify()
    s = df.to_markdown()
    return f"```\n{s}\n```"


if __name__ == "__main__":
    df = excelify()
    print(df.to_markdown())
