from pathlib import Path
import numpy as np
import pandas as pd
from run_analysis_scripts.tables_in_paper import generate_latex_rows, get_column_spec
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
    pre_wrapped=False,
    tab_col_sep="3pt",
    decimal_places=2,
    highlight_best=True,
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
        df, method_tuples, num_columns, extra_column_tuples
    )

    # Use enhanced column specification
    column_spec, multicolumn_line = get_enhanced_column_spec(GROUPS, extra_column_name)

    # Use enhanced header creation
    header_line = create_enhanced_header(GROUPS, DATASET_NAME_KEYS, extra_column_name)

    offset = 0
    if extra_column_name is not None:
        offset = 1

    caption_tex = f"""\\caption{{{caption}}}
\\label{{table:{label}}}
"""

    if CAPTION_TOP:
        caption_top, caption_bottom = caption_tex, ""
    else:
        caption_top, caption_bottom = "", caption_tex

    inner_table = f"""
{caption_top}
\\setlength{{\\tabcolsep}}{{{tab_col_sep}}}
\\begin{{tabular}}{{{column_spec}}}
\\hline
{multicolumn_line} \\\\
\\cmidrule(lr){{{2+offset}-{len(column_order)+offset}}}
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


def generate_enhanced_latex_rows(df, method_tuples, num_columns, extra_column_tuples):
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
            if len(row) == 1:
                row = row.values.tolist()[0]
                cells = method_column + extra_column
                for idx, x in enumerate(row[1:]):
                    cell = format_confidence_interval(x, max_values[idx])
                    cells.append(cell)
            else:
                if len(row) > 1:
                    print("REJECTED", row)
                cells = (
                    method_column
                    + extra_column
                    + ["\\textcolor{red}{??.?}"] * num_columns
                )
            latex_row = " & ".join(cells) + " \\\\"
        latex_rows += latex_row + "\n"
    return latex_rows


def format_confidence_interval(
    val_str, best_val, decimal_places=2, highlight_best=True
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
        if highlight_best and best_val is not None and abs(mean_val - best_val) < 1e-10:
            return f"\\textbf{{{mean_formatted}}}$_{{\\pm{sd_formatted}}}$"
        else:
            return f"{mean_formatted}$_{{\\pm{sd_formatted}}}$"

    except Exception as e:
        return val_str


def get_enhanced_column_spec(groups, extra_column_name):
    """Generate enhanced column specification."""
    # Method column
    spec_parts = ["l"]  # Left-aligned for method names

    # Extra column if needed
    if extra_column_name is not None:
        spec_parts.append("c")

    # Data columns - center aligned with some spacing
    num_data_cols = sum(len(group) for group in groups.values())
    spec_parts.extend(["c"] * num_data_cols)

    column_spec = "@{}" + "".join(spec_parts) + "@{}"

    # Create multicolumn line for group headers
    multicolumn_parts = ["Method"]
    if extra_column_name is not None:
        multicolumn_parts.append(extra_column_name)

    col_start = 2 if extra_column_name is not None else 1
    for group_name, group_cols in groups.items():
        col_end = col_start + len(group_cols) - 1
        if col_start == col_end:
            multicolumn_parts.append(f"\\textbf{{{group_name}}}")
        else:
            multicolumn_parts.append(
                f"\\multicolumn{{{len(group_cols)}}}{{c}}{{\\textbf{{{group_name}}}}}"
            )
        col_start = col_end + 1

    multicolumn_line = " & ".join(multicolumn_parts)

    return column_spec, multicolumn_line


def create_enhanced_header(groups, dataset_name_keys, extra_column_name):
    """Create enhanced header line."""
    header_parts = [""]  # Empty cell above method column

    if extra_column_name is not None:
        header_parts.append("")  # Empty cell above extra column

    for group in groups.values():
        for col in group:
            dataset_name = dataset_name_keys.get(col, col)
            header_parts.append(f"\\textsc{{{dataset_name}}}")

    return " & ".join(header_parts)


def generate_main_table_gemma_ci(df):
    caption = "\\textbf{Performance of our INSQUAD against existing approaches}, evaluated across nine distinct datasets on Gemma (2B). Our approach outperforms existing baselines on retrieval with the top-performing result for each dataset is highlighted in \\textbf{bold}."

    label = "main_table"
    method_tuples = (
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
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
        ("hline", "hline"),
        ("oracle_mpnet_gemma", "Oracle"),
    )

    prepared_df = prepare_table_dataframe(df, method_tuples)

    result = generate_latex_table(
        prepared_df,
        caption,
        label,
        method_tuples,
        extra_column_name=None,
        extra_column_tuples=None,
        decimal_places=2,
        highlight_best=True,
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
        decimal_places=2,
        highlight_best=True,
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
