from pathlib import Path
import pandas as pd
import numpy as np
from run_analysis_scripts.utils import extract_relevant_df
from scipy import stats


def generate_markdown_table(
    df, caption, method_tuples, extra_column_name=None, extra_column_tuples=None
):
    """
    Generate a markdown table from DataFrame with averages and 95% CI.

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    caption (str): Table caption
    method_tuples (tuple): Method name tuples (id, display_name)
    extra_column_name (str, optional): Name for grouping column
    extra_column_tuples (tuple, optional): Extra column mapping tuples

    Returns:
    str: Markdown table
    """
    df = extract_relevant_df(df.reset_index(), method_tuples)

    # Melt the DataFrame for easier manipulation
    df_melted = pd.melt(df, id_vars=["method"], var_name="dataset", value_name="value")
    df_melted = df_melted[~df_melted["dataset"].isin(["index", "Average"])]

    # Convert tuples to dictionaries for lookup
    method_lut = dict(method_tuples)
    extra_column_lut = dict(extra_column_tuples) if extra_column_tuples else {}

    # Apply lookup transformations
    df_melted["method_name"] = df_melted["method"].map(method_lut)

    if extra_column_name and extra_column_tuples:
        df_melted["extra_name"] = df_melted["method"].map(extra_column_lut)
        df_melted["group"] = df_melted["extra_name"]
        df_melted["name"] = df_melted["method_name"]
    else:
        df_melted["name"] = df_melted["method_name"]
        df_melted["group"] = df_melted["method_name"]

    # Calculate statistics by group
    stats_df = (
        df_melted.groupby(["group", "name"])
        .agg(
            mean_value=("value", "mean"),
            std_value=("value", "std"),
            count=("value", "count"),
        )
        .reset_index()
    )

    # Calculate 95% confidence interval
    stats_df["ci_95"] = stats_df.apply(
        lambda row: stats.t.ppf(0.975, row["count"] - 1)
        * row["std_value"]
        / np.sqrt(row["count"]),
        axis=1,
    )

    # Format for markdown table
    stats_df["formatted"] = stats_df.apply(
        lambda row: f"{row['mean_value']:.3f} ± {row['ci_95']:.3f}", axis=1
    )

    # Create markdown table
    markdown = f"## {caption}\n\n"

    if extra_column_name:
        # Pivot to get extra_column values as columns
        pivot_df = stats_df.pivot(index="name", columns="group", values="formatted")
        markdown += f"| Method | {' | '.join(pivot_df.columns)} |\n"
        markdown += f"|---|{' | '.join(['---' for _ in pivot_df.columns])}|\n"

        for idx, row in pivot_df.iterrows():
            markdown += f"| {idx} | {' | '.join(row.values)} |\n"
    else:
        # Simple table with just methods
        markdown += "| Method | Average (95% CI) |\n"
        markdown += "|---|---|\n"

        for idx, row in stats_df.iterrows():
            markdown += f"| {row['name']} | {row['formatted']} |\n"

    return markdown


def generate_retrieval_method_ablations_gemma(df):
    caption = "Effect of retrieval methods Gemma (2B)"
    method_tuples = (
        ("quaild_random_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_random_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_random_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_similar_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_similar_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_similar_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        ("quaild_random_fl_mpnet_gemma", "Random"),
        ("quaild_random_gc_mpnet_gemma", "Random"),
        ("quaild_random_ld_mpnet_gemma", "Random"),
        ("quaild_similar_fl_mpnet_gemma", "Similar"),
        ("quaild_similar_gc_mpnet_gemma", "Similar"),
        ("quaild_similar_ld_mpnet_gemma", "Similar"),
        ("quaild_comb_fl_mpnet_gemma_best", "Combinatorial"),
        ("quaild_comb_gc_mpnet_gemma_best", "Combinatorial"),
        ("quaild_comb_ld_mpnet_gemma_best", "Combinatorial"),
    )
    extra_column_name = "Retrieval"

    return generate_markdown_table(
        df, caption, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_retrieval_method_performance_gap_gemma(df):
    caption = "Performance Gap Gemma(2B)"
    method_tuples = (
        ("quaild_similar_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_similar_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_similar_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_combnt_fl_mpnet_gemma", "InSQuaD-FL (NT)"),
        ("quaild_combnt_gc_mpnet_gemma", "InSQuaD-GC (NT)"),
        ("quaild_combnt_ld_mpnet_gemma", "InSQuaD-LD (NT)"),
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
    )
    df = extract_relevant_df(df.reset_index(), method_tuples)

    # Get all dataset columns (excluding "method" and "Average")
    dataset_columns = [
        col for col in df.columns if col not in ["method", "Average", "index"]
    ]

    if dataset_columns:
        # Create markdown table header
        md_table = "\n## Per-dataset Performance Comparison\n\n"
        md_table += "| Dataset | Similar | Combinatorial | % Increase Combinatorial |\n"
        md_table += "|---------|------------|-----------------------------------|-----------------------------|\n"

        # Initialize variables to calculate averages
        similar_total = 0
        comb_best_total = 0
        comb_best_inc_total = 0

        # Add rows for each dataset
        for dataset in dataset_columns:
            # Get max performance per method type for this dataset
            similar_ds_max = df[df["method"].str.contains("similar")][dataset].max()
            combnt_ds_max = df[df["method"].str.contains("combnt")][dataset].max()
            comb_best_ds_max = df[df["method"].str.contains("comb_.*_best")][
                dataset
            ].max()

            # Calculate percentage increases for this dataset
            combnt_ds_increase = (
                (combnt_ds_max - similar_ds_max) / similar_ds_max
            ) * 100
            comb_best_ds_increase = (
                (comb_best_ds_max - similar_ds_max) / similar_ds_max
            ) * 100

            # Add row to markdown table
            md_table += f"| {dataset} | {similar_ds_max:.4f} | {comb_best_ds_max:.4f} | {comb_best_ds_increase:.2f}% |\n"

            # Accumulate values for average calculation
            similar_total += similar_ds_max
            comb_best_total += comb_best_ds_max
            comb_best_inc_total += comb_best_ds_increase

        # Calculate averages
        dataset_count = len(dataset_columns)
        similar_avg = similar_total / dataset_count
        comb_best_avg = comb_best_total / dataset_count
        comb_best_inc_avg = comb_best_inc_total / dataset_count

        # Add a separator row
        md_table += "|---------|------------|-----------------------------------|-----------------------------|\n"

        # Add average row
        md_table += f"| **Average** | **{similar_avg:.4f}** | **{comb_best_avg:.4f}** | **{comb_best_inc_avg:.2f}%** |\n"

    return f"## {caption} \n\n{md_table}"


def generate_only_quality_performance_gap_gemma(df):
    caption = "Performance Gap λ > 0 Gemma (2B)"
    method_tuples = (
        ("quaild_comb_fl_mpnet_gemma_lambda_0", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_0", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_0", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma_lambda_025", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_025", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_025", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma_lambda_1", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_1", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_1", "InSQuaD-LD"),
    )
    df = extract_relevant_df(df.reset_index(), method_tuples)

    # Get all dataset columns (excluding "method" and "Average")
    dataset_columns = [
        col for col in df.columns if col not in ["method", "Average", "index"]
    ]

    if dataset_columns:
        # Create markdown table header
        md_table = "\n## Per-dataset Performance Comparison\n\n"
        md_table += "| Dataset | λ = 0 | λ > 0 | % Increase λ > 0 |\n"
        md_table += "|---------|------------|-----------------|--------------------|\n"

        # Initialize variables to calculate averages
        lambda0_total = 0
        lambdaGT0_total = 0
        increase_total = 0

        # Add rows for each dataset
        for dataset in dataset_columns:
            # Get max performance for lambda=0 methods for this dataset
            lambda0_ds_max = df[df["method"].str.contains("lambda_0$")][dataset].max()

            # Get max performance for lambda>0 methods for this dataset
            # This includes lambda_025, lambda=0.5 (no suffix), and lambda_1
            lambdaGT0_methods = df["method"].str.contains("lambda_025|lambda_1") | (
                df["method"].str.contains("quaild_comb_")
                & ~df["method"].str.contains("lambda_")
            )
            lambdaGT0_ds_max = df[lambdaGT0_methods][dataset].max()

            # Calculate percentage increase
            increase = ((lambdaGT0_ds_max - lambda0_ds_max) / lambda0_ds_max) * 100

            # Add row to markdown table
            md_table += f"| {dataset} | {lambda0_ds_max:.4f} | {lambdaGT0_ds_max:.4f} | {increase:.2f}% |\n"

            # Accumulate values for average calculation
            lambda0_total += lambda0_ds_max
            lambdaGT0_total += lambdaGT0_ds_max
            increase_total += increase

        # Calculate averages
        dataset_count = len(dataset_columns)
        lambda0_avg = lambda0_total / dataset_count
        lambdaGT0_avg = lambdaGT0_total / dataset_count
        increase_avg = increase_total / dataset_count

        # Add a separator row
        md_table += "|---------|------------|-----------------|--------------------|\n"

        # Add average row
        md_table += f"| **Average** | **{lambda0_avg:.4f}** | **{lambdaGT0_avg:.4f}** | **{increase_avg:.2f}%** |\n"

    return f"## {caption} \n\n{md_table}"


def generate_annotation_budget_ablations_gemma(df):
    caption = "Effects of annotation budget Gemma (2B) λ = 0.5"
    method_tuples = (
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma_100", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_100", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_100", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        ("quaild_comb_fl_mpnet_gemma", "18"),
        ("quaild_comb_gc_mpnet_gemma", "18"),
        ("quaild_comb_ld_mpnet_gemma", "18"),
        ("quaild_comb_fl_mpnet_gemma_100", "100"),
        ("quaild_comb_gc_mpnet_gemma_100", "100"),
        ("quaild_comb_ld_mpnet_gemma_100", "100"),
    )
    extra_column_name = "Budget"

    return generate_markdown_table(
        df, caption, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_qd_tradeoff_ablations_gemma(df):
    caption = "Effects of λ on Gemma (2B) (Quality-Diversity tradeoff)"
    method_tuples = (
        ("quaild_comb_fl_mpnet_gemma_lambda_0", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_0", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_0", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma_lambda_025", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_025", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_025", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma_lambda_1", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_1", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_1", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        ("quaild_comb_fl_mpnet_gemma_lambda_0", "0"),
        ("quaild_comb_gc_mpnet_gemma_lambda_0", "0"),
        ("quaild_comb_ld_mpnet_gemma_lambda_0", "0"),
        ("quaild_comb_fl_mpnet_gemma_lambda_025", "0.25"),
        ("quaild_comb_gc_mpnet_gemma_lambda_025", "0.25"),
        ("quaild_comb_ld_mpnet_gemma_lambda_025", "0.25"),
        ("quaild_comb_fl_mpnet_gemma", "0.5"),
        ("quaild_comb_gc_mpnet_gemma", "0.5"),
        ("quaild_comb_ld_mpnet_gemma", "0.5"),
        ("quaild_comb_fl_mpnet_gemma_lambda_1", "1"),
        ("quaild_comb_gc_mpnet_gemma_lambda_1", "1"),
        ("quaild_comb_ld_mpnet_gemma_lambda_1", "1"),
    )
    extra_column_name = "λ"

    return generate_markdown_table(
        df, caption, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_model_size_ablations(df):
    caption = "Effects of model size"
    method_tuples = (
        # gemma
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("oracle_mpnet_gemma", "Oracle"),
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        # gemma7b
        ("zeroshot_mpnet_gemma7b", "Zeroshot"),
        ("random_mpnet_gemma7b", "Random"),
        ("oracle_mpnet_gemma7b", "Oracle"),
        ("quaild_comb_fl_mpnet_gemma7b", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma7b", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma7b", "InSQuaD-LD"),
        # davinci2
        ("zeroshot_mpnet_davinci2", "Zeroshot"),
        ("random_mpnet_davinci2", "Random"),
        ("oracle_mpnet_davinci2", "Oracle"),
        ("quaild_comb_fl_mpnet_davinci2", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_davinci2", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_davinci2", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        # gemma
        ("zeroshot_mpnet_gemma", "gemma2b"),
        ("random_mpnet_gemma", "gemma2b"),
        ("oracle_mpnet_gemma", "gemma2b"),
        ("quaild_comb_fl_mpnet_gemma", "gemma2b"),
        ("quaild_comb_gc_mpnet_gemma", "gemma2b"),
        ("quaild_comb_ld_mpnet_gemma", "gemma2b"),
        # gemma7b
        ("zeroshot_mpnet_gemma7b", "gemma7b"),
        ("random_mpnet_gemma7b", "gemma7b"),
        ("oracle_mpnet_gemma7b", "gemma7b"),
        ("quaild_comb_fl_mpnet_gemma7b", "gemma7b"),
        ("quaild_comb_gc_mpnet_gemma7b", "gemma7b"),
        ("quaild_comb_ld_mpnet_gemma7b", "gemma7b"),
        # davinci2
        ("zeroshot_mpnet_davinci2", "davinci2-175b"),
        ("random_mpnet_davinci2", "davinci2-175b"),
        ("oracle_mpnet_davinci2", "davinci2-175b"),
        ("quaild_comb_fl_mpnet_davinci2", "davinci2-175b"),
        ("quaild_comb_gc_mpnet_davinci2", "davinci2-175b"),
        ("quaild_comb_ld_mpnet_davinci2", "davinci2-175b"),
    )
    extra_column_name = "Model"

    return generate_markdown_table(
        df, caption, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_main_figure_gemma(df):
    caption = "Downstream evaluation on Gemma (2B)"
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
        ("quaild_combnt_fl_mpnet_gemma", "InSQuaD-FL (NT)"),
        ("quaild_combnt_gc_mpnet_gemma", "InSQuaD-GC (NT)"),
        ("quaild_combnt_ld_mpnet_gemma", "InSQuaD-LD (NT)"),
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
    )

    return generate_markdown_table(df, caption, method_tuples)


def run_markdown_generation(df_path):
    """Generate all markdown tables and write to file."""
    TABLES_TO_GENERATE = {
        "main_table_gemma": generate_main_figure_gemma,
        "model_size_effect": generate_model_size_ablations,
        "qd_tradeoff_gemma": generate_qd_tradeoff_ablations_gemma,
        "annotation_budget_effect_gemma": generate_annotation_budget_ablations_gemma,
        "retrieval_method_effect_gemma": generate_retrieval_method_ablations_gemma,
        "retrieval_method_performance_gap_gemma": generate_retrieval_method_performance_gap_gemma,
        "only_quality_performance_gap_gemma": generate_only_quality_performance_gap_gemma,
    }

    df = pd.read_csv(df_path)

    all_tables = []
    for name, generator_fn in TABLES_TO_GENERATE.items():
        table_md = generator_fn(df)
        all_tables.append(f"# {name}\n\n{table_md}\n\n")

    return "\n".join(all_tables)


# For demo purposes
if __name__ == "__main__":
    BASE_PATH = Path("./artifacts/markdowns")
    BASE_PATH.mkdir(parents=True, exist_ok=True)

    # Generate tables
    tables_md = run_markdown_generation("./artifacts/tables/all.csv")
    with open(BASE_PATH / "all_tables.md", "w", encoding="utf-8") as f:
        f.write(tables_md)

    print("Tables generated successfully!")
