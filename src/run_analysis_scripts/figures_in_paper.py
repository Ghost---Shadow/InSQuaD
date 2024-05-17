from pathlib import Path
from run_analysis_scripts.excelify import excelify
from run_analysis_scripts.utils import extract_relevant_df, generate_best_row
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd


def generate_bar_plot(
    df,
    caption,
    label,
    method_tuples,
    extra_column_name=None,
    extra_column_tuples=None,
    hue="extra_name",
    subplot_adjust=0.0,
    y_label=None,
):
    df = extract_relevant_df(df.reset_index(), method_tuples)

    # Convert tuples to dictionaries for easy lookup
    method_lut = dict(method_tuples)
    extra_column_lut = dict(extra_column_tuples) if extra_column_tuples else {}

    # Apply lookup transformations
    df["method_name"] = df["method"].map(method_lut)
    if extra_column_name and extra_column_tuples:
        df["extra_name"] = df["method"].map(extra_column_lut)
        df["name"] = df["method_name"] + " " + df["extra_name"]
    else:
        df["name"] = df["method_name"]

    # Set the Seaborn theme
    sns.set_theme("paper")

    # Create a figure with specific size
    plt.figure(figsize=(6, 6))
    plt.tight_layout()
    plt.subplots_adjust(subplot_adjust)
    plt.xlim(0, 1)

    # Sort the DataFrame for plotting
    sorted_df = df.sort_values(by="Average", ascending=True)
    # print(sorted_df)

    # Create a bar plot
    if extra_column_name is not None:
        if hue == "extra_name":
            y = "method_name"
        else:
            y = "extra_name"
        sns.barplot(x="Average", y=y, hue=hue, data=sorted_df)
    else:
        sns.barplot(x="Average", y="method_name", data=sorted_df)

    plt.xlabel("Accuracy")
    plt.ylabel(y_label)
    if extra_column_name is not None:
        plt.legend(title=extra_column_name)

    # Set the plot title
    plt.title(caption)

    # Determine the output path
    output_path = Path("artifacts/diagrams")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(output_path / f"{label}.png")
    plt.close()
    plt.clf()


def generate_dataset_wise_bar_plot(
    df,
    caption,
    label,
    method_tuples,
    extra_column_name,
    extra_column_tuples,
    subplot_adjust=0.0,
):
    df = extract_relevant_df(df.reset_index(), method_tuples)

    df_melted = pd.melt(df, id_vars=["method"])
    df_melted = df_melted[~df_melted["dataset"].isin(["index", "Average"])]

    # Convert tuples to dictionaries for easy lookup
    method_lut = dict(method_tuples)
    extra_column_lut = dict(extra_column_tuples) if extra_column_tuples else {}

    # Apply lookup transformations
    df_melted["method_name"] = df_melted["method"].map(method_lut)
    if extra_column_name and extra_column_tuples:
        df_melted["extra_name"] = df_melted["method"].map(extra_column_lut)
        df_melted["name"] = df_melted["method_name"] + " " + df_melted["extra_name"]
    else:
        df_melted["name"] = df_melted["method_name"]

    # Set the Seaborn theme
    sns.set_theme("paper")

    # Create a figure with specific size
    plt.figure(figsize=(6, 6))
    plt.tight_layout()
    plt.subplots_adjust(subplot_adjust)
    plt.xlim(0, 1)

    sorted_df = df_melted.sort_values(
        by=["dataset", "extra_name", "value"], ascending=True
    )

    sns.barplot(x="value", y="dataset", hue="extra_name", data=sorted_df)

    plt.xlabel("Accuracy")
    plt.ylabel(None)
    plt.legend(title=extra_column_name)

    # Set the plot title
    plt.title(caption)

    # Determine the output path
    output_path = Path("artifacts/diagrams")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(output_path / f"{label}.png")
    plt.close()
    plt.clf()


def generate_retrieval_method_ablations_gemma(df):
    caption = "Effect of retrieval methods Gemma (2B)"
    label = "retrieval_method_ablations"
    method_tuples = (
        ("quaild_random_fl_mpnet_gemma", "QuailD-FL"),
        ("quaild_random_gc_mpnet_gemma", "QuailD-GC"),
        ("quaild_similar_fl_mpnet_gemma", "QuailD-FL"),
        ("quaild_similar_gc_mpnet_gemma", "QuailD-GC"),
        ("quaild_gain_fl_mpnet_gemma_best", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma_best", "QuailD-GC"),
    )
    extra_column_tuples = (
        ("quaild_random_fl_mpnet_gemma", "Random"),
        ("quaild_random_gc_mpnet_gemma", "Random"),
        ("quaild_similar_fl_mpnet_gemma", "Similar"),
        ("quaild_similar_gc_mpnet_gemma", "Similar"),
        ("quaild_gain_fl_mpnet_gemma_best", "Combinatorial"),
        ("quaild_gain_gc_mpnet_gemma_best", "Combinatorial"),
    )
    extra_column_name = "Retrieval"
    result = generate_dataset_wise_bar_plot(
        df,
        caption,
        label,
        method_tuples,
        extra_column_name,
        extra_column_tuples,
        subplot_adjust=0.15,
    )

    return result


def generate_annotation_budget_ablations_gemma(df):
    caption = "Effects of annotation budget Gemma (2B) $\\lambda = 0.5$"
    label = "budget_ablations"
    method_tuples = (
        ("quaild_gain_fl_mpnet_gemma", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma", "QuailD-GC"),
        ("quaild_gain_fl_mpnet_gemma_100", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma_100", "QuailD-GC"),
    )

    extra_column_tuples = (
        ("quaild_gain_fl_mpnet_gemma", "18"),
        ("quaild_gain_gc_mpnet_gemma", "18"),
        ("quaild_gain_fl_mpnet_gemma_100", "100"),
        ("quaild_gain_gc_mpnet_gemma_100", "100"),
    )

    extra_column_name = "Budget"
    result = generate_dataset_wise_bar_plot(
        df,
        caption,
        label,
        method_tuples,
        extra_column_name,
        extra_column_tuples,
        subplot_adjust=0.15,
    )

    return result


def generate_qd_tradeoff_ablations_gemma(df):
    caption = "Effects of $\\lambda$ on Gemma (2B) (Quality-Diversity tradeoff)"
    label = "qd_tradeoff"
    method_tuples = (
        ("quaild_gain_fl_mpnet_gemma_lambda_0", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma_lambda_0", "QuailD-GC"),
        ("quaild_gain_fl_mpnet_gemma_lambda_025", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma_lambda_025", "QuailD-GC"),
        ("quaild_gain_fl_mpnet_gemma", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma", "QuailD-GC"),
        ("quaild_gain_fl_mpnet_gemma_lambda_1", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma_lambda_1", "QuailD-GC"),
    )
    extra_column_tuples = (
        ("quaild_gain_fl_mpnet_gemma_lambda_0", "0"),
        ("quaild_gain_gc_mpnet_gemma_lambda_0", "0"),
        ("quaild_gain_fl_mpnet_gemma_lambda_025", "0.25"),
        ("quaild_gain_gc_mpnet_gemma_lambda_025", "0.25"),
        ("quaild_gain_fl_mpnet_gemma", "0.5"),
        ("quaild_gain_gc_mpnet_gemma", "0.5"),
        ("quaild_gain_fl_mpnet_gemma_lambda_1", "1"),
        ("quaild_gain_gc_mpnet_gemma_lambda_1", "1"),
    )

    extra_column_name = "$\\lambda$"
    result = generate_dataset_wise_bar_plot(
        df,
        caption,
        label,
        method_tuples,
        extra_column_name,
        extra_column_tuples,
        subplot_adjust=0.15,
    )

    return result


def generate_model_size_ablations(df):
    caption = "Effects of model size"
    label = "model_size"
    method_tuples = (
        # gemma
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("oracle_mpnet_gemma", "Oracle"),
        ("quaild_gain_fl_mpnet_gemma_best", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma_best", "QuailD-GC"),
        # gemma7b
        ("zeroshot_mpnet_gemma7b", "Zeroshot"),
        ("random_mpnet_gemma7b", "Random"),
        ("oracle_mpnet_gemma7b", "Oracle"),
        ("quaild_gain_fl_mpnet_gemma7b", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma7b", "QuailD-GC"),
        # davinci2
        ("zeroshot_mpnet_davinci2", "Zeroshot"),
        ("random_mpnet_davinci2", "Random"),
        ("oracle_mpnet_davinci2", "Oracle"),
        ("quaild_gain_fl_mpnet_davinci2", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_davinci2", "QuailD-GC"),
    )
    extra_column_tuples = (
        # gemma
        ("zeroshot_mpnet_gemma", "gemma2b"),
        ("random_mpnet_gemma", "gemma2b"),
        ("oracle_mpnet_gemma", "gemma2b"),
        ("quaild_gain_fl_mpnet_gemma_best", "gemma2b"),
        ("quaild_gain_gc_mpnet_gemma_best", "gemma2b"),
        # gemma7b
        ("zeroshot_mpnet_gemma7b", "gemma7b"),
        ("random_mpnet_gemma7b", "gemma7b"),
        ("oracle_mpnet_gemma7b", "gemma7b"),
        ("quaild_gain_fl_mpnet_gemma7b", "gemma7b"),
        ("quaild_gain_gc_mpnet_gemma7b", "gemma7b"),
        # davinci2
        ("zeroshot_mpnet_davinci2", "davinci2-175b"),
        ("random_mpnet_davinci2", "davinci2-175b"),
        ("oracle_mpnet_davinci2", "davinci2-175b"),
        ("quaild_gain_fl_mpnet_davinci2", "davinci2-175b"),
        ("quaild_gain_gc_mpnet_davinci2", "davinci2-175b"),
    )

    extra_column_name = "Model"
    result = generate_bar_plot(
        df,
        caption,
        label,
        method_tuples,
        extra_column_name,
        extra_column_tuples,
        hue="method_name",
        subplot_adjust=0.2,
    )

    return result


def generate_main_figure_gemma(df):
    caption = "Downstream evaluation on Gemma (2B)"
    label = "gemma_results"
    method_tuples = (
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("oracle_mpnet_gemma", "Oracle"),
        ("diversity_mpnet_gemma", "Diversity"),
        ("leastconfidence_mpnet_gemma", "Least Confidence"),
        ("mfl_mpnet_gemma", "MFL"),
        ("gc_mpnet_gemma", "GC"),
        ("votek_mpnet_gemma", "Vote-K"),
        ("ideal_mpnet_gemma", "IDEAL"),
        ("quaild_nt_fl_mpnet_gemma", "QuailD-FL (NT)"),
        ("quaild_nt_gc_mpnet_gemma", "QuailD-GC (NT)"),
        ("quaild_gain_fl_mpnet_gemma_best", "QuailD-FL"),
        ("quaild_gain_gc_mpnet_gemma_best", "QuailD-GC"),
    )

    extra_column_tuples = None
    extra_column_name = None
    result = generate_bar_plot(
        df,
        caption,
        label,
        method_tuples,
        extra_column_name,
        extra_column_tuples,
        subplot_adjust=0.2,
    )

    return result


if __name__ == "__main__":
    FIGURES_TO_GENERATE = {
        "main_table_gemma": generate_main_figure_gemma,
        "model_size_effect": generate_model_size_ablations,
        "qd_tradeoff_gemma": generate_qd_tradeoff_ablations_gemma,
        "annotation_budget_effect_gemma": generate_annotation_budget_ablations_gemma,
        "retrieval_method_effect_gemma": generate_retrieval_method_ablations_gemma,
    }
    BASE_PATH = Path("./artifacts/tables")
    BASE_PATH.mkdir(parents=True, exist_ok=True)

    df = excelify()
    df = df.reset_index()
    df = generate_best_row(df)
    df.to_csv(BASE_PATH / "all.csv")

    for file_name, fn in tqdm(FIGURES_TO_GENERATE.items()):
        fn(df)
