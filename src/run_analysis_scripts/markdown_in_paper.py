from pathlib import Path
import pandas as pd
import numpy as np
from run_analysis_scripts.utils import extract_relevant_df
from scipy import stats


def generate_markdown_table(
    df,
    caption,
    method_tuples,
    extra_column_name=None,
    extra_column_tuples=None,
    exclude_columns=None,
    column_order=None,
):
    """
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    caption (str): Table caption
    method_tuples (tuple): Method name tuples (id, display_name)
    extra_column_name (str, optional): Name for grouping column
    extra_column_tuples (tuple, optional): Extra column mapping tuples
    exclude_columns (list, optional): List of column names to exclude from the table
    column_order (list, optional): List specifying the order of dataset columns

    Returns:
    str: Markdown table
    """
    import pandas as pd
    import numpy as np
    from scipy import stats

    # Initialize exclude_columns if None
    if exclude_columns is None:
        exclude_columns = ["seed"]

    if column_order is None:
        column_order = [
            "mrpc",
            "sst5",
            "mnli",
            "dbpedia",
            "rte",
            "hellaswag",
            "mwoz",
            "geoq",
            "xsum",
        ]

    df = extract_relevant_df(df.reset_index(), method_tuples)

    # Melt the DataFrame for easier manipulation
    df_melted = pd.melt(df, id_vars=["method"], var_name="dataset", value_name="value")

    # Exclude specified columns and default columns to exclude
    default_exclude = ["index", "Average", "seed"]
    all_exclude = list(set(default_exclude + exclude_columns))
    df_melted = df_melted[~df_melted["dataset"].isin(all_exclude)]

    # Convert tuples to dictionaries for lookup
    method_lut = dict(method_tuples)
    extra_column_lut = dict(extra_column_tuples) if extra_column_tuples else {}

    # Apply lookup transformations
    df_melted["method_name"] = df_melted["method"].map(method_lut)

    # Apply extra column mapping if provided
    if extra_column_name and extra_column_tuples:
        df_melted["extra_name"] = df_melted["method"].map(extra_column_lut)
        df_melted["group"] = df_melted["extra_name"]
        df_melted["name"] = df_melted["method_name"]
    else:
        df_melted["name"] = df_melted["method_name"]
        df_melted["group"] = df_melted["method"]  # Use method ID as group

    # Create a table with datasets as columns and methods as rows
    # First get unique datasets
    datasets = df_melted["dataset"].unique()

    # For each method and dataset, calculate the statistics
    result_dict = {}

    # Group by both method and dataset to get the statistics
    for method_id, method_name in method_tuples:
        method_data = {}
        for dataset in datasets:
            # Filter data for this method and dataset
            filtered_data = df_melted[
                (df_melted["method"] == method_id) & (df_melted["dataset"] == dataset)
            ]
            if not filtered_data.empty:
                values = filtered_data["value"].values
                mean_val = np.mean(values)
                # Handle the case where we have only one value (no std)
                if len(values) > 1:
                    std_val = np.std(values, ddof=1)
                    ci_95 = (
                        stats.t.ppf(0.975, len(values) - 1)
                        * std_val
                        / np.sqrt(len(values))
                    )
                    method_data[dataset] = f"{mean_val:.4f} ± {ci_95:.4f}"
                else:
                    error_margin = 0.0
                    method_data[dataset] = f"{mean_val:.4f} ± {error_margin:.4f}"
            else:
                method_data[dataset] = "-"

        # Store results with method name
        if extra_column_name and extra_column_tuples:
            # Use extra_column_lut to get the group name
            group_name = extra_column_lut.get(method_id, "")
            # Create combined key with group and method name
            combined_key = f"{group_name}: {method_name}"
            result_dict[combined_key] = method_data
        else:
            result_dict[method_name] = method_data

    # Generate the markdown table
    markdown = f"## {caption}\n\n"

    # Create header row with enforced column order if provided
    if column_order:
        # Filter to include only columns that exist in the data
        datasets_list = [col for col in column_order if col in datasets]
        # Add any columns from the data that aren't in column_order at the end
        missing_cols = [col for col in datasets if col not in column_order]
        datasets_list.extend(missing_cols)
    else:
        datasets_list = list(datasets)

    # Add extra column header if provided
    if extra_column_name:
        header = [extra_column_name, "Method"] + datasets_list
    else:
        header = ["Method"] + datasets_list

    markdown += "| " + " | ".join(header) + " |\n"
    markdown += "| " + " | ".join(["---" for _ in header]) + " |\n"

    # Group methods by extra column if provided
    if extra_column_name and extra_column_tuples:
        # Get unique groups
        groups = set(extra_column_lut.values())

        # Organize method names by group
        grouped_methods = {}
        for method_id, method_name in method_tuples:
            group = extra_column_lut.get(method_id, "")
            if group not in grouped_methods:
                grouped_methods[group] = []
            grouped_methods[group].append(method_name)

        # Generate rows group by group
        for group in sorted(grouped_methods.keys()):
            for method_name in grouped_methods[group]:
                combined_key = f"{group}: {method_name}"
                dataset_values = result_dict[combined_key]

                row_values = [
                    group,
                    method_name,
                ]  # First column is group, second is method
                for dataset in datasets_list:
                    row_values.append(dataset_values.get(dataset, "-"))
                markdown += "| " + " | ".join(row_values) + " |\n"
    else:
        # No grouping, just list methods
        method_names = list(result_dict.keys())
        for method_name in method_names:
            dataset_values = result_dict[method_name]
            row_values = [method_name]
            for dataset in datasets_list:
                row_values.append(dataset_values.get(dataset, "-"))
            markdown += "| " + " | ".join(row_values) + " |\n"

    return markdown


def generate_retrieval_method_ablations_gemma(df):
    caption = "Effect of retrieval methods Gemma (2B)"
    method_tuples = (
        # ("quaild_random_fl_mpnet_gemma", "InSQuaD-FL"),
        # ("quaild_random_gc_mpnet_gemma", "InSQuaD-GC"),
        # ("quaild_random_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_similar_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_similar_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_similar_ld_mpnet_gemma", "InSQuaD-LD"),
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        # ("quaild_random_fl_mpnet_gemma", "Random"),
        # ("quaild_random_gc_mpnet_gemma", "Random"),
        # ("quaild_random_ld_mpnet_gemma", "Random"),
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

    # Reset index and extract relevant dataframe
    df_reset = df.reset_index()
    df_relevant = extract_relevant_df(df_reset, method_tuples)

    # Get all dataset columns (excluding "method", "Average", "index", and "seed")
    dataset_columns = [
        col
        for col in df_relevant.columns
        if col not in ["method", "Average", "index", "seed"]
    ]

    if dataset_columns:
        # Create markdown table header
        md_table = "\n## Per-dataset Performance Comparison (interval = sem * stats.t.ppf((1 + confidence=0.95) / 2, n - 1)) \n\n"
        md_table += "| Dataset | Similar (95% CI) | Combinatorial (95% CI) | % Increase Combinatorial (95% CI) |\n"
        md_table += "|---------|-----------------|------------------------|----------------------------------|\n"

        # Initialize variables to calculate averages
        similar_values = []
        comb_best_values = []
        increase_values = []

        # Process each dataset
        for dataset in dataset_columns:
            # Group by method and seed, then get values for confidence interval calculation
            similar_values_ds = []
            combnt_values_ds = []
            comb_best_values_ds = []

            # Get values for each method type across all seeds
            similar_methods = [m for m, _ in method_tuples if "similar" in m]
            combnt_methods = [m for m, _ in method_tuples if "combnt" in m]
            comb_best_methods = [
                m for m, _ in method_tuples if "comb_" in m and "_best" in m
            ]

            # Extract all values for similar methods
            similar_df = df_relevant[df_relevant["method"].isin(similar_methods)]
            if not similar_df.empty:
                grouped_similar = (
                    similar_df.groupby(["method", "seed"])[dataset].max().reset_index()
                )
                similar_max_per_seed = grouped_similar.groupby("seed")[dataset].max()
                similar_values_ds = similar_max_per_seed.tolist()

            # Extract all values for combinatorial best methods
            comb_best_df = df_relevant[df_relevant["method"].isin(comb_best_methods)]
            if not comb_best_df.empty:
                grouped_comb_best = (
                    comb_best_df.groupby(["method", "seed"])[dataset]
                    .max()
                    .reset_index()
                )
                comb_best_max_per_seed = grouped_comb_best.groupby("seed")[
                    dataset
                ].max()
                comb_best_values_ds = comb_best_max_per_seed.tolist()

            # Calculate means
            similar_mean = np.mean(similar_values_ds) if similar_values_ds else 0
            comb_best_mean = np.mean(comb_best_values_ds) if comb_best_values_ds else 0

            # Calculate percentage increases for each seed
            increase_values_ds = []
            if (
                len(similar_values_ds) == len(comb_best_values_ds)
                and len(similar_values_ds) > 0
            ):
                for s_val, cb_val in zip(similar_values_ds, comb_best_values_ds):
                    if s_val > 0:
                        increase = ((cb_val - s_val) / s_val) * 100
                        increase_values_ds.append(increase)

            increase_mean = np.mean(increase_values_ds) if increase_values_ds else 0

            # Calculate confidence intervals (95%)
            similar_ci = (
                calculate_confidence_interval(similar_values_ds)
                if len(similar_values_ds) > 1
                else (similar_mean, similar_mean)
            )
            comb_best_ci = (
                calculate_confidence_interval(comb_best_values_ds)
                if len(comb_best_values_ds) > 1
                else (comb_best_mean, comb_best_mean)
            )
            increase_ci = (
                calculate_confidence_interval(increase_values_ds)
                if len(increase_values_ds) > 1
                else (increase_mean, increase_mean)
            )

            # Format confidence intervals
            similar_ci_str = (
                f"{similar_mean:.4f} ({similar_ci[0]:.4f}-{similar_ci[1]:.4f})"
            )
            comb_best_ci_str = (
                f"{comb_best_mean:.4f} ({comb_best_ci[0]:.4f}-{comb_best_ci[1]:.4f})"
            )
            increase_ci_str = (
                f"{increase_mean:.2f}% ({increase_ci[0]:.2f}%-{increase_ci[1]:.2f}%)"
            )

            # Add row to markdown table
            md_table += f"| {dataset} | {similar_ci_str} | {comb_best_ci_str} | {increase_ci_str} |\n"

            # Store values for overall average calculation
            similar_values.append(similar_mean)
            comb_best_values.append(comb_best_mean)
            increase_values.append(increase_mean)

        # Calculate overall averages
        similar_avg = np.mean(similar_values) if similar_values else 0
        comb_best_avg = np.mean(comb_best_values) if comb_best_values else 0
        increase_avg = np.mean(increase_values) if increase_values else 0

        # Calculate overall confidence intervals
        similar_overall_ci = (
            calculate_confidence_interval(similar_values)
            if len(similar_values) > 1
            else (similar_avg, similar_avg)
        )
        comb_best_overall_ci = (
            calculate_confidence_interval(comb_best_values)
            if len(comb_best_values) > 1
            else (comb_best_avg, comb_best_avg)
        )
        increase_overall_ci = (
            calculate_confidence_interval(increase_values)
            if len(increase_values) > 1
            else (increase_avg, increase_avg)
        )

        # Format overall confidence intervals
        similar_overall_ci_str = f"{similar_avg:.4f} ({similar_overall_ci[0]:.4f}-{similar_overall_ci[1]:.4f})"
        comb_best_overall_ci_str = f"{comb_best_avg:.4f} ({comb_best_overall_ci[0]:.4f}-{comb_best_overall_ci[1]:.4f})"
        increase_overall_ci_str = f"{increase_avg:.2f}% ({increase_overall_ci[0]:.2f}%-{increase_overall_ci[1]:.2f}%)"

        # Add a separator row
        md_table += "|---------|-----------------|------------------------|----------------------------------|\n"
        # Add average row with confidence intervals
        md_table += f"| **Average** | **{similar_overall_ci_str}** | **{comb_best_overall_ci_str}** | **{increase_overall_ci_str}** |\n"

    return f"## {caption} \n\n{md_table}"


def calculate_confidence_interval(values, confidence=0.95):
    """
    Calculate confidence interval for a list of values.

    Parameters:
    values (list): List of numerical values
    confidence (float): Confidence level (default: 0.95 for 95% confidence)

    Returns:
    tuple: (lower_bound, upper_bound) of the confidence interval
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard Error of the Mean
    interval = sem * stats.t.ppf((1 + confidence) / 2, n - 1)  # t-value for CI
    return (mean - interval, mean + interval)


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

    # Create a mapping from method name to method type
    method_to_type = {method: type_name for method, type_name in method_tuples}

    # Reset index and extract relevant dataframe
    df_reset = df.reset_index()
    df_relevant = extract_relevant_df(df_reset, method_tuples)

    # Make a explicit copy to avoid the SettingWithCopyWarning
    df_relevant = df_relevant.copy()

    # Add method_type column based on the mapping
    df_relevant.loc[:, "method_type"] = df_relevant["method"].map(method_to_type)

    # Get all dataset columns (excluding "method", "method_type", "Average", "index", and "seed")
    dataset_columns = [
        col
        for col in df_relevant.columns
        if col not in ["method", "method_type", "Average", "index", "seed"]
    ]

    if dataset_columns:
        # Create markdown table header
        md_table = "\n## Per-dataset Performance Comparison (interval = sem * stats.t.ppf((1 + confidence=0.95) / 2, n - 1))\n\n"
        md_table += "| Method Type | Dataset | λ = 0 (95% CI) | λ > 0 (95% CI) | % Increase λ > 0 (95% CI) |\n"
        md_table += "|------------|---------|----------------|----------------|---------------------------|\n"

        # Initialize dictionaries to store values for final average calculation
        method_type_results = {}

        # First, process each method type
        for method_type in sorted(df_relevant["method_type"].unique()):
            # Filter dataframe for current method type
            df_method_type = df_relevant[df_relevant["method_type"] == method_type]

            # Initialize storage for this method type's results
            method_type_results[method_type] = {
                "lambda0_values": [],
                "lambdaGT0_values": [],
                "increase_values": [],
            }

            # Process each dataset for this method type
            for dataset in dataset_columns:
                # Lists to store values across seeds for confidence interval calculation
                lambda0_values_ds = []
                lambdaGT0_values_ds = []
                increase_values_ds = []

                # Get methods for lambda=0 and lambda>0 for this method type
                lambda0_methods = [
                    m
                    for m, t in method_tuples
                    if t == method_type and m.endswith("lambda_0")
                ]

                lambdaGT0_methods = [
                    m
                    for m, t in method_tuples
                    if t == method_type
                    and (
                        m.endswith("lambda_025")
                        or m.endswith("lambda_1")
                        or (
                            ("quaild_comb_fl_mpnet_gemma" == m and t == "InSQuaD-FL")
                            or ("quaild_comb_gc_mpnet_gemma" == m and t == "InSQuaD-GC")
                            or ("quaild_comb_ld_mpnet_gemma" == m and t == "InSQuaD-LD")
                        )
                    )
                ]

                # Filter and group by seed for lambda=0 methods
                lambda0_df = df_method_type[
                    df_method_type["method"].isin(lambda0_methods)
                ]
                if not lambda0_df.empty and "seed" in lambda0_df.columns:
                    # Group by seed and get max performance for each seed
                    lambda0_by_seed = (
                        lambda0_df.groupby("seed")[dataset].max().reset_index()
                    )
                    lambda0_values_ds = lambda0_by_seed[dataset].tolist()

                # Filter and group by seed for lambda>0 methods
                lambdaGT0_df = df_method_type[
                    df_method_type["method"].isin(lambdaGT0_methods)
                ]
                if not lambdaGT0_df.empty and "seed" in lambdaGT0_df.columns:
                    # Group by seed and get max performance for each seed
                    lambdaGT0_by_seed = (
                        lambdaGT0_df.groupby("seed")[dataset].max().reset_index()
                    )
                    lambdaGT0_values_ds = lambdaGT0_by_seed[dataset].tolist()

                # Calculate means across seeds
                lambda0_mean = np.mean(lambda0_values_ds) if lambda0_values_ds else 0
                lambdaGT0_mean = (
                    np.mean(lambdaGT0_values_ds) if lambdaGT0_values_ds else 0
                )

                # Calculate percentage increase for each seed pair
                if (
                    len(lambda0_values_ds) == len(lambdaGT0_values_ds)
                    and len(lambda0_values_ds) > 0
                ):
                    for l0_val, lgt0_val in zip(lambda0_values_ds, lambdaGT0_values_ds):
                        if l0_val > 0:
                            increase = ((lgt0_val - l0_val) / l0_val) * 100
                            increase_values_ds.append(increase)

                # Calculate mean increase
                increase_mean = np.mean(increase_values_ds) if increase_values_ds else 0

                # Calculate confidence intervals (95%)
                lambda0_ci = (
                    calculate_confidence_interval(lambda0_values_ds)
                    if len(lambda0_values_ds) > 1
                    else (lambda0_mean, lambda0_mean)
                )
                lambdaGT0_ci = (
                    calculate_confidence_interval(lambdaGT0_values_ds)
                    if len(lambdaGT0_values_ds) > 1
                    else (lambdaGT0_mean, lambdaGT0_mean)
                )
                increase_ci = (
                    calculate_confidence_interval(increase_values_ds)
                    if len(increase_values_ds) > 1
                    else (increase_mean, increase_mean)
                )

                # Format confidence intervals
                lambda0_ci_str = (
                    f"{lambda0_mean:.4f} ({lambda0_ci[0]:.4f}-{lambda0_ci[1]:.4f})"
                )
                lambdaGT0_ci_str = f"{lambdaGT0_mean:.4f} ({lambdaGT0_ci[0]:.4f}-{lambdaGT0_ci[1]:.4f})"
                increase_ci_str = f"{increase_mean:.2f}% ({increase_ci[0]:.2f}%-{increase_ci[1]:.2f}%)"

                # Add row to markdown table
                md_table += f"| {method_type} | {dataset} | {lambda0_ci_str} | {lambdaGT0_ci_str} | {increase_ci_str} |\n"

                # Store values for method type average calculation
                method_type_results[method_type]["lambda0_values"].append(lambda0_mean)
                method_type_results[method_type]["lambdaGT0_values"].append(
                    lambdaGT0_mean
                )
                method_type_results[method_type]["increase_values"].append(
                    increase_mean
                )

            # Calculate method type averages
            lambda0_avg = (
                np.mean(method_type_results[method_type]["lambda0_values"])
                if method_type_results[method_type]["lambda0_values"]
                else 0
            )
            lambdaGT0_avg = (
                np.mean(method_type_results[method_type]["lambdaGT0_values"])
                if method_type_results[method_type]["lambdaGT0_values"]
                else 0
            )
            increase_avg = (
                np.mean(method_type_results[method_type]["increase_values"])
                if method_type_results[method_type]["increase_values"]
                else 0
            )

            # Calculate method type confidence intervals
            lambda0_mt_ci = (
                calculate_confidence_interval(
                    method_type_results[method_type]["lambda0_values"]
                )
                if len(method_type_results[method_type]["lambda0_values"]) > 1
                else (lambda0_avg, lambda0_avg)
            )
            lambdaGT0_mt_ci = (
                calculate_confidence_interval(
                    method_type_results[method_type]["lambdaGT0_values"]
                )
                if len(method_type_results[method_type]["lambdaGT0_values"]) > 1
                else (lambdaGT0_avg, lambdaGT0_avg)
            )
            increase_mt_ci = (
                calculate_confidence_interval(
                    method_type_results[method_type]["increase_values"]
                )
                if len(method_type_results[method_type]["increase_values"]) > 1
                else (increase_avg, increase_avg)
            )

            # Format method type confidence intervals
            lambda0_mt_ci_str = (
                f"{lambda0_avg:.4f} ({lambda0_mt_ci[0]:.4f}-{lambda0_mt_ci[1]:.4f})"
            )
            lambdaGT0_mt_ci_str = f"{lambdaGT0_avg:.4f} ({lambdaGT0_mt_ci[0]:.4f}-{lambdaGT0_mt_ci[1]:.4f})"
            increase_mt_ci_str = f"{increase_avg:.2f}% ({increase_mt_ci[0]:.2f}%-{increase_mt_ci[1]:.2f}%)"

            # Add method type average row
            md_table += f"| **{method_type} Avg** | | **{lambda0_mt_ci_str}** | **{lambdaGT0_mt_ci_str}** | **{increase_mt_ci_str}** |\n"
            md_table += "|------------|---------|----------------|----------------|---------------------------|\n"

        # Calculate overall averages across all method types and datasets
        all_lambda0_values = [
            val for mt in method_type_results.values() for val in mt["lambda0_values"]
        ]
        all_lambdaGT0_values = [
            val for mt in method_type_results.values() for val in mt["lambdaGT0_values"]
        ]
        all_increase_values = [
            val for mt in method_type_results.values() for val in mt["increase_values"]
        ]

        lambda0_overall_avg = np.mean(all_lambda0_values) if all_lambda0_values else 0
        lambdaGT0_overall_avg = (
            np.mean(all_lambdaGT0_values) if all_lambdaGT0_values else 0
        )
        increase_overall_avg = (
            np.mean(all_increase_values) if all_increase_values else 0
        )

        # Calculate overall confidence intervals
        lambda0_overall_ci = (
            calculate_confidence_interval(all_lambda0_values)
            if len(all_lambda0_values) > 1
            else (lambda0_overall_avg, lambda0_overall_avg)
        )
        lambdaGT0_overall_ci = (
            calculate_confidence_interval(all_lambdaGT0_values)
            if len(all_lambdaGT0_values) > 1
            else (lambdaGT0_overall_avg, lambdaGT0_overall_avg)
        )
        increase_overall_ci = (
            calculate_confidence_interval(all_increase_values)
            if len(all_increase_values) > 1
            else (increase_overall_avg, increase_overall_avg)
        )

        # Format overall confidence intervals
        lambda0_overall_ci_str = f"{lambda0_overall_avg:.4f} ({lambda0_overall_ci[0]:.4f}-{lambda0_overall_ci[1]:.4f})"
        lambdaGT0_overall_ci_str = f"{lambdaGT0_overall_avg:.4f} ({lambdaGT0_overall_ci[0]:.4f}-{lambdaGT0_overall_ci[1]:.4f})"
        increase_overall_ci_str = f"{increase_overall_avg:.2f}% ({increase_overall_ci[0]:.2f}%-{increase_overall_ci[1]:.2f}%)"

        # Add overall average row
        md_table += f"| **Overall Avg** | | **{lambda0_overall_ci_str}** | **{lambdaGT0_overall_ci_str}** | **{increase_overall_ci_str}** |\n"

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
    tables_md = run_markdown_generation("./artifacts/tables/all_merged.csv")
    with open(BASE_PATH / "all_tables.md", "w", encoding="utf-8") as f:
        f.write(tables_md)

    print("Tables generated successfully!")
