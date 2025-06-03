from pathlib import Path
import numpy as np
import pandas as pd
from run_analysis_scripts.utils import (
    DATASET_NAME_KEYS,
    GROUPS,
    dictify,
    prepare_table_dataframe,
)
from tqdm import tqdm

CAPTION_TOP = True


def generate_latex_table(
    df,
    caption,
    label,
    method_tuples,
    extra_column_name=None,
    extra_column_tuples=None,
    include_rank=False,
    pre_wrapped=False,
    tab_col_sep="3pt",
    decimal_places=2,
):
    # Reset the index to make 'method' a regular column
    df = df.reset_index()

    if extra_column_tuples is not None:
        get_key = lambda x: x[0]
        assert set(map(get_key, extra_column_tuples)) == set(
            map(get_key, method_tuples)
        )
        assert extra_column_name is not None

    # Ensure all required columns are present in the DataFrame
    expected_columns = [col for group in GROUPS.values() for col in group]
    column_order = ["method"] + expected_columns
    for column in expected_columns:
        if column.lower() not in df.columns.str.lower():
            df[column] = np.nan  # Add missing columns as NaN

    # Reorder DataFrame to match the exact column order needed
    df = df[column_order]

    # Use enhanced latex row generation for confidence intervals
    num_columns = len(expected_columns)
    latex_rows = generate_enhanced_latex_rows(
        df, method_tuples, num_columns, extra_column_tuples, include_rank
    )

    # Use enhanced column specification
    column_spec, multicolumn_line = get_enhanced_column_spec(GROUPS, extra_column_name, include_rank)

    # Use enhanced header creation
    header_line = create_enhanced_header(GROUPS, DATASET_NAME_KEYS, extra_column_name, include_rank)

    offset = 0
    if extra_column_name is not None:
        offset += 1

    caption_tex = f"""\\caption{{{caption}}}
\\label{{table:{label}}}
"""

    if CAPTION_TOP:
        caption_top, caption_bottom = caption_tex, ""
    else:
        caption_top, caption_bottom = "", caption_tex

    # Calculate the cmidrule range - exclude method, extra column, and rank column
    cmidrule_start = 2 + offset
    cmidrule_end = len(column_order) + offset - (1 if include_rank else 0)
    
    inner_table = f"""
{caption_top}
\\setlength{{\\tabcolsep}}{{{tab_col_sep}}}
\\begin{{tabular}}{{{column_spec}}}
\\hline
{multicolumn_line} \\\\
\\cmidrule(lr){{{cmidrule_start}-{cmidrule_end}}}
{header_line} \\\\
\\hline
{latex_rows}
\\hline
\\end{{tabular}}
{caption_bottom}
"""

    latex_template = f"""
\\begin{{table}}[H]
\\centering
\\small
{inner_table}
\\end{{table}}
"""
    if pre_wrapped:
        return latex_template
    else:
        return inner_table


def compute_average_ranks(df):
    """Compute average rank for each method across all datasets with confidence intervals."""
    df_copy = df.copy()
    
    def safe_float_convert(x):
        try:
            return float(x.split()[0])
        except (ValueError, IndexError, AttributeError):
            return float('-inf')
    
    # Convert confidence intervals to numeric values for ranking
    numeric_df = df_copy.copy()
    for col in df_copy.columns:
        if col != 'method':
            numeric_df[col] = df_copy[col].apply(safe_float_convert)
    
    # Calculate ranks for each dataset (column)
    ranks = {}
    for col in numeric_df.columns:
        if col != 'method':
            # Rank in descending order (higher values get better ranks)
            col_ranks = numeric_df[col].rank(method='average', ascending=False)
            for idx, method in enumerate(numeric_df['method']):
                if method not in ranks:
                    ranks[method] = []
                ranks[method].append(col_ranks.iloc[idx])
    
    # Calculate average rank and confidence interval for each method
    avg_ranks = {}
    for method in ranks:
        valid_ranks = [r for r in ranks[method] if not np.isnan(r) and r != float('inf')]
        if valid_ranks and len(valid_ranks) > 1:
            mean_rank = np.mean(valid_ranks)
            from run_analysis_scripts.utils import calculate_confidence_interval
            low, high = calculate_confidence_interval(valid_ranks)
            # Calculate standard deviation from confidence interval
            # Assuming 95% CI, so SD ≈ (upper - lower) / (2 * 1.96)
            sd = (high - low) / (2 * 1.96)
            avg_ranks[method] = f"{mean_rank:.1f}$_{{\\pm{sd:.1f}}}$"
        elif valid_ranks:
            mean_rank = np.mean(valid_ranks)
            avg_ranks[method] = f"{mean_rank:.1f} ERROR"
        else:
            avg_ranks[method] = "-"
    
    return avg_ranks


def generate_enhanced_latex_rows(df, method_tuples, num_columns, extra_column_tuples, include_rank=False):
    latex_rows = ""

    # Set all values to -inf where the method contains 'oracle'
    df_copy = df.copy()

    def safe_float_convert(x):
        try:
            return float(x.split()[0])
        except (ValueError, IndexError, AttributeError):
            return x

    df_copy = df_copy.map(safe_float_convert)
    for i in range(1, num_columns + 1):
        mask = df_copy["method"].str.contains("oracle")
        df_copy.loc[mask, df_copy.columns[i]] = 0.0

        mask = df_copy["method"].str.contains("hline")
        df_copy.loc[mask, df_copy.columns[i]] = 0.0

    max_values = [df_copy.iloc[:, i].max() for i in range(1, num_columns + 1)]
    
    # Calculate average ranks if needed
    avg_ranks = {}
    if include_rank:
        avg_ranks = compute_average_ranks(df)

    for method, method_print_name in method_tuples:
        if method == "hline":
            latex_row = "\hline"
        else:
            row = df[df["method"] == method]
            method_column = [method_print_name]
            extra_column = []
            if extra_column_tuples is not None:
                extra_column_lut = dictify(extra_column_tuples)
                extra_column = [extra_column_lut[method]]
            
            rank_column = []
            if include_rank:
                if method in avg_ranks:
                    rank_column = [avg_ranks[method]]
                else:
                    rank_column = ["-"]
            
            if len(row) == 1:
                row = row.values.tolist()[0]
                cells = method_column + extra_column
                for idx, x in enumerate(row[1:]):
                    cell = format_confidence_interval(x, max_values[idx])
                    cells.append(cell)
                # Add rank column at the end
                cells.extend(rank_column)
            else:
                if len(row) > 1:
                    print("REJECTED", row)
                cells = (
                    method_column
                    + extra_column
                    + ["\\textcolor{red}{??.?}"] * num_columns
                    + rank_column
                )
            latex_row = " & ".join(cells) + " \\\\"
        latex_rows += latex_row + "\n"
    return latex_rows


def format_confidence_interval(
    val_str, best_val, decimal_places=2
):
    """Format a confidence interval string with proper LaTeX formatting."""
    try:
        # Parse the string: "0.5153 (0.4777, 0.5529)"
        parts = val_str.split("(")
        if len(parts) != 2:
            return val_str

        mean_str = parts[0].strip()
        ci_str = parts[1].rstrip(")")

        mean_val = float(mean_str)
        ci_parts = [float(x.strip()) for x in ci_str.split(",")]

        # Calculate standard deviation from confidence interval
        # Assuming 95% CI, so SD ≈ (upper - lower) / (2 * 1.96)
        sd = (ci_parts[1] - ci_parts[0]) / (2 * 1.96)

        # Format with specified decimal places
        mean_formatted = f"{mean_val:.{decimal_places}f}"
        sd_formatted = f"{sd:.{decimal_places}f}"

        # Highlight if this is the best value
        if best_val is not None and abs(mean_val - best_val) < 1e-10:
            return f"\\textbf{{{mean_formatted}}}$_{{\\pm{sd_formatted}}}$"
        else:
            return f"{mean_formatted}$_{{\\pm{sd_formatted}}}$"

    except Exception:
        return val_str


def get_enhanced_column_spec(groups, extra_column_name, include_rank=False):
    """Generate enhanced column specification."""
    # Method column
    spec_parts = ["l"]  # Left-aligned for method names

    # Extra column if needed
    if extra_column_name is not None:
        spec_parts.append("c")

    # Data columns - center aligned with some spacing
    num_data_cols = sum(len(group) for group in groups.values())
    spec_parts.extend(["c"] * num_data_cols)
    
    # Rank column at the end if needed
    if include_rank:
        spec_parts.append("c")

    column_spec = "@{}" + "".join(spec_parts) + "@{}"

    # Create multicolumn line for group headers
    multicolumn_parts = ["Method"]
    if extra_column_name is not None:
        multicolumn_parts.append(extra_column_name)

    col_start = 2 + (1 if extra_column_name is not None else 0)
    for group_name, group_cols in groups.items():
        col_end = col_start + len(group_cols) - 1
        if col_start == col_end:
            multicolumn_parts.append(f"\\textbf{{{group_name}}}")
        else:
            multicolumn_parts.append(
                f"\\multicolumn{{{len(group_cols)}}}{{c}}{{\\textbf{{{group_name}}}}}"
            )
        col_start = col_end + 1
    
    # Add rank column at the end
    if include_rank:
        multicolumn_parts.append("Rank")

    multicolumn_line = " & ".join(multicolumn_parts)

    return column_spec, multicolumn_line


def create_enhanced_header(groups, dataset_name_keys, extra_column_name, include_rank=False):
    """Create enhanced header line."""
    header_parts = [""]  # Empty cell above method column

    if extra_column_name is not None:
        header_parts.append("")  # Empty cell above extra column

    for group in groups.values():
        for col in group:
            dataset_name = dataset_name_keys.get(col, col)
            header_parts.append(f"\\textsc{{{dataset_name}}}")
    
    if include_rank:
        header_parts.append("")  # Empty cell above rank column

    return " & ".join(header_parts)


def compute_best_rows(df):
    method_tuples = (
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("oracle_mpnet_gemma", "Oracle"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_0", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_0", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_0", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_025", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_025", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_025", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_1", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_1", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_1", "InSQuaD-LD"),
    )
    prepared_df = prepare_table_dataframe(df, method_tuples)
    

    # Define the method groups for each InSQuaD variant
    fl_methods = [
        "quaild_comb_fl_mpnet_gemma_lambda_0",
        "quaild_comb_fl_mpnet_gemma_lambda_025",
        "quaild_comb_fl_mpnet_gemma",
        "quaild_comb_fl_mpnet_gemma_lambda_1",
    ]

    gc_methods = [
        "quaild_comb_gc_mpnet_gemma_lambda_0",
        "quaild_comb_gc_mpnet_gemma_lambda_025",
        "quaild_comb_gc_mpnet_gemma",
        "quaild_comb_gc_mpnet_gemma_lambda_1",
    ]

    ld_methods = [
        "quaild_comb_ld_mpnet_gemma_lambda_0",
        "quaild_comb_ld_mpnet_gemma_lambda_025",
        "quaild_comb_ld_mpnet_gemma",
        "quaild_comb_ld_mpnet_gemma_lambda_1",
    ]

    # Get column names (excluding 'method' column)
    columns = [col for col in prepared_df.columns if col != "method"]

    # Function to extract numeric value from string like "0.5153 (0.4777, 0.5529)"
    def extract_value(cell_value):
        if pd.isna(cell_value) or cell_value == "-":
            return float("-inf")
        # Extract the first number before the parentheses
        return float(str(cell_value).split(" ")[0])

    # Create best rows for each method group
    best_rows = []

    for method_group, group_name in [
        (fl_methods, "quaild_comb_fl_mpnet_gemma_best"),
        (gc_methods, "quaild_comb_gc_mpnet_gemma_best"),
        (ld_methods, "quaild_comb_ld_mpnet_gemma_best"),
    ]:
        best_row = {"method": group_name}
        

        for col in columns:
            # Find the best performing method for this column
            best_value = float("-inf")
            best_cell_content = None

            for method in method_group:
                # Find the row with this method
                method_row = prepared_df[prepared_df["method"] == method]
                if not method_row.empty:
                    cell_value = method_row[col].iloc[0]
                    numeric_value = extract_value(cell_value)

                    if numeric_value > best_value:
                        best_value = numeric_value
                        best_cell_content = cell_value

            best_row[col] = best_cell_content if best_cell_content is not None else "-"

        best_rows.append(best_row)

    # Create DataFrame from best rows and append to prepared_df
    best_df = pd.DataFrame(best_rows)

    return best_df


def generate_main_table_gemma_ci(df):
    caption = "\\textbf{Performance of our INSQUAD against existing approaches}, evaluated across nine distinct datasets on Gemma (2B). Our approach outperforms existing baselines on retrieval with the top-performing result for each dataset is highlighted in \\textbf{bold}."

    label = "main_table"
    head_method_tuples = (
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("diversity_mpnet_gemma", "Diversity"),
        ("leastconfidence_mpnet_gemma", "Least Confidence"),
        ("mfl_mpnet_gemma", "MFL"),
        ("gc_mpnet_gemma", "GC"),
        ("votek_mpnet_gemma", "Vote-K"),
        ("ideal_mpnet_gemma", "IDEAL"),
        ("hline", "hline"),
        ("quaild_combnt_fl_mpnet_gemma", "InSQuaD-FL (NT)"),
        ("quaild_combnt_gc_mpnet_gemma", "InSQuaD-GC (NT)"),
        ("quaild_combnt_ld_mpnet_gemma", "InSQuaD-LD (NT)"),
        ("hline", "hline"),
    )
    best_method_tuples = (
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
    )
    tail_method_tuples = (
        ("hline", "hline"),
        ("oracle_mpnet_gemma", "Oracle"),
    )
    head_and_tail = (*head_method_tuples, *tail_method_tuples)
    all_method_tuples = (*head_method_tuples, *best_method_tuples, *tail_method_tuples)

    best_rows = compute_best_rows(df)

    prepared_df = prepare_table_dataframe(df, head_and_tail)

    result_df = pd.concat([prepared_df, best_rows], ignore_index=True)

    result = generate_latex_table(
        result_df,
        caption,
        label,
        all_method_tuples,
        extra_column_name=None,
        extra_column_tuples=None,
        include_rank=True,
        decimal_places=2,
    )

    return result


def generate_qd_tradeoff_ablations_gemma_ci(df):
    caption = "Effects of $\\lambda$ on Gemma (2B) (Quality-Diversity tradeoff)"
    label = "qd_tradeoff"
    method_tuples = (
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("oracle_mpnet_gemma", "Oracle"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_0", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_0", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_0", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_025", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_025", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_025", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_1", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_lambda_1", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_lambda_1", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        ("zeroshot_mpnet_gemma", ""),
        ("random_mpnet_gemma", ""),
        ("oracle_mpnet_gemma", ""),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_0", "0"),
        ("quaild_comb_gc_mpnet_gemma_lambda_0", "0"),
        ("quaild_comb_ld_mpnet_gemma_lambda_0", "0"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_025", "0.25"),
        ("quaild_comb_gc_mpnet_gemma_lambda_025", "0.25"),
        ("quaild_comb_ld_mpnet_gemma_lambda_025", "0.25"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma", "0.5"),
        ("quaild_comb_gc_mpnet_gemma", "0.5"),
        ("quaild_comb_ld_mpnet_gemma", "0.5"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_lambda_1", "1"),
        ("quaild_comb_gc_mpnet_gemma_lambda_1", "1"),
        ("quaild_comb_ld_mpnet_gemma_lambda_1", "1"),
    )

    extra_column_name = "$\\lambda$"

    prepared_df = prepare_table_dataframe(df, method_tuples)

    result = generate_latex_table(
        prepared_df,
        caption,
        label,
        method_tuples,
        extra_column_name=extra_column_name,
        extra_column_tuples=extra_column_tuples,
        include_rank=True,
        decimal_places=2,
    )

    return result


if __name__ == "__main__":
    BASE_PATH = Path("./artifacts/tables")
    BASE_PATH.mkdir(parents=True, exist_ok=True)

    TABLES_TO_GENERATE = {
        "main_table_gemma_ci": generate_main_table_gemma_ci,
        "qd_tradeoff_gemma_ci": generate_qd_tradeoff_ablations_gemma_ci,
    }

    df = pd.read_csv("./artifacts/tables/all_merged.csv")

    for file_name, fn in tqdm(TABLES_TO_GENERATE.items()):
        with open(BASE_PATH / f"{file_name}.tex", "w") as f:
            f.write(fn(df))

    print("Tables generated successfully!")
