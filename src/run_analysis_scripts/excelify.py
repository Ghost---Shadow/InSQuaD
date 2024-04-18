import os
import json
import pandas as pd


def excelify():
    BASE_DIR = "artifacts"
    data = []

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
                                    data.append(
                                        {
                                            "method": method,
                                            "seed": seed,
                                            "dataset": dataset,
                                            "accuracy": result_data["accuracy"],
                                        }
                                    )

    df = pd.DataFrame(data)

    if not df.empty:
        df_grouped = (
            df.groupby(["method", "dataset"]).agg({"accuracy": "mean"}).reset_index()
        )
        pivot_df = df_grouped.pivot(
            index="method", columns="dataset", values="accuracy"
        )

        # Calculate the average accuracy for each method
        pivot_df["Average"] = pivot_df.mean(axis=1)

        # Sort by average accuracy in descending order
        pivot_df = pivot_df.sort_values(by="Average", ascending=False)

        return pivot_df
    else:
        return None


def excelify_for_discord():
    df = excelify()
    if df is not None:
        s = df.to_markdown()
        return f"```\n{s}\n```"
    return "No data available."


if __name__ == "__main__":
    df = excelify()
    if df is not None:
        print(df.to_markdown())
    else:
        print("No data available.")
