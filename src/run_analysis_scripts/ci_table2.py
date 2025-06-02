from pathlib import Path
import numpy as np
import pandas as pd
from run_analysis_scripts.tables_in_paper import generate_latex_rows, get_column_spec
from run_analysis_scripts.utils import (
    DATASET_NAME_KEYS,
    GROUPS,
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
    latex_rows = generate_enhanced_latex_rows(
        df,
        method_tuples,
        expected_columns,
        extra_column_tuples,
        highlight_best,
        decimal_places,
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


def generate_enhanced_latex_rows(
    df,
    method_tuples,
    expected_columns,
    extra_column_tuples,
    highlight_best,
    decimal_places,
):
    """Generate LaTeX rows with enhanced formatting for confidence intervals."""
    rows = []

    # Extract numeric values for finding best results (excluding Oracle)
    numeric_data = {}
    for col in expected_columns:
        numeric_data[col] = []
        for _, row in df.iterrows():
            # Skip Oracle rows when collecting data for best value calculation
            if row["method"] == "Oracle":
                continue

            val_str = str(row[col])
            if pd.isna(row[col]) or val_str == "nan":
                numeric_data[col].append(np.nan)
            else:
                # Extract the mean value (before the parentheses)
                try:
                    mean_val = float(val_str.split("(")[0].strip())
                    numeric_data[col].append(mean_val)
                except:
                    numeric_data[col].append(np.nan)

    # Find best values for each column (excluding Oracle)
    best_values = {}
    if highlight_best:
        for col in expected_columns:
            vals = [v for v in numeric_data[col] if not np.isnan(v)]
            if vals:
                best_values[col] = max(vals)

    # Generate rows
    for method_key, method_display in method_tuples:
        # Handle horizontal lines
        if method_key == "hline":
            rows.append("\\hline")
            continue

        row_data = df[df["method"] == method_display]
        if row_data.empty:
            continue

        row = row_data.iloc[0]
        latex_row = [f"\\textsc{{{method_display}}}"]

        # Add extra column if needed
        if extra_column_tuples is not None:
            extra_val = next(
                (val for key, val in extra_column_tuples if key == method_key), ""
            )
            latex_row.append(str(extra_val))

        # Add data columns with enhanced formatting
        for col in expected_columns:
            val_str = str(row[col])
            if pd.isna(row[col]) or val_str == "nan":
                formatted_val = "\\hline"  # Use \hline for missing values
            else:
                # Don't highlight Oracle as best, even if it has the highest value
                should_highlight = highlight_best and method_display != "Oracle"
                formatted_val = format_confidence_interval(
                    val_str, best_values.get(col), decimal_places, should_highlight
                )
            latex_row.append(formatted_val)

        rows.append(" & ".join(latex_row) + " \\\\")

    return "\n".join(rows)


def format_confidence_interval(val_str, best_val, decimal_places, highlight_best):
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

    prepared_df, _, _ = prepare_table_dataframe(
        df,
        method_tuples,
        extra_column_name=None,
        extra_column_tuples=None,
        exclude_columns=None,
    )

    # Rename method column to match expected format
    if "Method" in prepared_df.columns:
        prepared_df["method"] = prepared_df["Method"]
        prepared_df = prepared_df.drop("Method", axis=1)

    result = generate_latex_table(
        prepared_df,
        caption,
        label,
        method_tuples,
        extra_column_name=None,
        extra_column_tuples=None,
        decimal_places=2,
        highlight_best=True,  # Enable highlighting of best values
    )

    return result


if __name__ == "__main__":
    BASE_PATH = Path("./artifacts/tables")
    BASE_PATH.mkdir(parents=True, exist_ok=True)

    TABLES_TO_GENERATE = {
        "main_table_gemma_ci": generate_main_table_gemma_ci,
    }

    df = pd.read_csv("./artifacts/tables/all_merged.csv")

    for file_name, fn in tqdm(TABLES_TO_GENERATE.items()):
        with open(BASE_PATH / f"{file_name}.tex", "w") as f:
            f.write(fn(df))

    print("Tables generated successfully!")
