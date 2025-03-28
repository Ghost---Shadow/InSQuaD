from pathlib import Path
import pandas as pd
import numpy as np
from run_analysis_scripts.utils import dictify, extract_relevant_df
from tqdm import tqdm

# Keeping the dataset name mappings and groups from the original code
DATASET_NAME_KEYS = {
    "dbpedia": "DBpedia",
    "dummy": "DummyDataset",
    "dummy_hotpot_qa_with_q": "DummyHotpotQaWithQDataset",
    "hellaswag": "HellaSwag",
    "hotpot_qa": "HotpotQaDataset",
    "hotpot_qa_with_q": "HotpotQaWithQDataset",
    "mnli": "MNLI",
    "mrpc": "MRPC",
    "rte": "RTE",
    "sst2": "SST2",
    "sst5": "SST5",
    "wiki_multihop_qa": "WikiMultihopQaDataset",
    "wiki_multihop_qa_with_q": "WikiMultihopQaWithQDataset",
    "xsum": "Xsum",
    "mwoz": "MWoZ",
    "geoq": "GeoQ",
}

GROUPS = {
    "Classification": ["mrpc", "sst5", "mnli", "dbpedia", "rte"],
    "Multi-Choice": ["hellaswag"],
    "Dialogue": ["mwoz"],
    "Generation": ["geoq", "xsum"],
}


def create_header_row(expected_columns, extra_column_name=None):
    """Create the header rows for the markdown table."""
    # First row with column names
    header1 = ["Method"]
    if extra_column_name:
        header1.append(extra_column_name)

    # Add dataset names
    for col in expected_columns:
        header1.append(f"**{DATASET_NAME_KEYS.get(col, col)}**")

    # Second row with separators
    header2 = ["---"] * len(header1)

    return [header1, header2]


def generate_markdown_rows(full_df, method_tuples, extra_column_tuples=None):
    """Generate the data rows for the markdown table."""
    rows = []
    df = extract_relevant_df(full_df, method_tuples)

    # Find max values for each column (excluding 'oracle' methods)
    df_copy = df.copy()
    columns = df_copy.columns[1:]  # Skip the 'method' column
    for col in columns:
        mask = df_copy["method"].str.contains("oracle")
        df_copy.loc[mask, col] = 0.0

    max_values = {col: df_copy[col].max() for col in columns}

    for method, method_print_name in method_tuples:
        if method == "hline":
            # For markdown, we can skip horizontal lines or use a separator
            continue

        row = df[df["method"] == method]
        cells = [method_print_name]

        # Add extra column if provided
        if extra_column_tuples is not None:
            extra_column_lut = dictify(extra_column_tuples)
            cells.append(extra_column_lut.get(method, ""))

        if len(row) == 1:
            row_data = row.iloc[0]
            for col in columns:
                value = row_data[col]
                if pd.notna(value):
                    # Bold the max values
                    if value == max_values.get(col, -float("inf")):
                        cells.append(f"**{value*100:.1f}**")
                    else:
                        cells.append(f"{value*100:.1f}")
                else:
                    cells.append("??")
        else:
            # Handle missing or multiple rows
            cells.extend(["??" for _ in columns])

        rows.append(cells)

    return rows


def generate_markdown_table(
    df, title, method_tuples, extra_column_name=None, extra_column_tuples=None
):
    """Generate a complete markdown table."""
    # Reset the index to make 'method' a regular column
    df = df.reset_index()

    if extra_column_tuples is not None:
        assert extra_column_name is not None

    # Ensure all required columns are present
    expected_columns = [col for group in GROUPS.values() for col in group]
    column_order = ["method"] + expected_columns
    for column in expected_columns:
        if column.lower() not in df.columns.str.lower():
            df[column] = np.nan  # Add missing columns as NaN

    # Reorder DataFrame to match column order
    df = df[column_order]

    # Create header rows
    header_rows = create_header_row(GROUPS, extra_column_name)

    # Create data rows
    data_rows = generate_markdown_rows(df, method_tuples, extra_column_tuples)

    # Combine all rows into a markdown table
    markdown_table = f"### {title}\n\n"
    all_rows = header_rows + data_rows

    for row in all_rows:
        markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return markdown_table


def generate_retrieval_method_ablations_gemma(df):
    """Generate markdown table for retrieval method ablations."""
    title = "Effect of retrieval methods Gemma (2B)"
    method_tuples = (
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("oracle_mpnet_gemma", "Oracle"),
        ("hline", "hline"),
        ("quaild_random_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_random_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_random_ld_mpnet_gemma", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_similar_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_similar_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_similar_ld_mpnet_gemma", "InSQuaD-LD"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        ("zeroshot_mpnet_gemma", ""),
        ("random_mpnet_gemma", ""),
        ("oracle_mpnet_gemma", ""),
        ("hline", "hline"),
        ("quaild_random_fl_mpnet_gemma", "Random"),
        ("quaild_random_gc_mpnet_gemma", "Random"),
        ("quaild_random_ld_mpnet_gemma", "Random"),
        ("hline", "hline"),
        ("quaild_similar_fl_mpnet_gemma", "Similar"),
        ("quaild_similar_gc_mpnet_gemma", "Similar"),
        ("quaild_similar_ld_mpnet_gemma", "Similar"),
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_best", "Combinatorial"),
        ("quaild_comb_gc_mpnet_gemma_best", "Combinatorial"),
        ("quaild_comb_ld_mpnet_gemma_best", "Combinatorial"),
    )
    extra_column_name = "Retrieval"

    return generate_markdown_table(
        df, title, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_annotation_budget_ablations_gemma(df):
    """Generate markdown table for annotation budget ablations."""
    title = "Effects of annotation budget Gemma (2B) λ = 0.5"
    method_tuples = (
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("oracle_mpnet_gemma", "Oracle"),
        # budget 18
        ("hline", "hline"),
        ("random_mpnet_gemma", "Random"),
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        # budget 100
        ("hline", "hline"),
        ("random_mpnet_gemma_100", "Random"),
        ("quaild_comb_fl_mpnet_gemma_100", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_100", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_100", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        # Zeroshot
        ("zeroshot_mpnet_gemma", ""),
        ("oracle_mpnet_gemma", ""),
        # budget 18
        ("hline", "hline"),
        ("random_mpnet_gemma", "18"),
        ("quaild_comb_fl_mpnet_gemma", "18"),
        ("quaild_comb_gc_mpnet_gemma", "18"),
        ("quaild_comb_ld_mpnet_gemma", "18"),
        # budget 100
        ("hline", "hline"),
        ("random_mpnet_gemma_100", "100"),
        ("quaild_comb_fl_mpnet_gemma_100", "100"),
        ("quaild_comb_gc_mpnet_gemma_100", "100"),
        ("quaild_comb_ld_mpnet_gemma_100", "100"),
    )
    extra_column_name = "Budget"

    return generate_markdown_table(
        df, title, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_qd_tradeoff_ablations_gemma(df):
    """Generate markdown table for quality-diversity tradeoff ablations."""
    title = "Effects of λ on Gemma (2B) (Quality-Diversity tradeoff)"
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
    extra_column_name = "λ"

    return generate_markdown_table(
        df, title, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_model_size_ablations(df):
    """Generate markdown table for model size ablations."""
    title = "Downstream evaluation on different model sizes"
    method_tuples = (
        # gemma
        ("zeroshot_mpnet_gemma", "Zeroshot"),
        ("random_mpnet_gemma", "Random"),
        ("oracle_mpnet_gemma", "Oracle"),
        ("votek_mpnet_gemma", "Vote-K"),
        ("ideal_mpnet_gemma", "IDEAL"),
        ("quaild_comb_fl_mpnet_gemma", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma", "InSQuaD-LD"),
        # gemma7b
        ("hline", "hline"),
        ("zeroshot_mpnet_gemma7b", "Zeroshot"),
        ("random_mpnet_gemma7b", "Random"),
        ("oracle_mpnet_gemma7b", "Oracle"),
        ("votek_mpnet_gemma7b", "Vote-K"),
        ("ideal_mpnet_gemma7b", "IDEAL"),
        ("quaild_comb_fl_mpnet_gemma7b", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma7b", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma7b", "InSQuaD-LD"),
        # davinci2
        ("hline", "hline"),
        ("zeroshot_mpnet_davinci2", "Zeroshot"),
        ("random_mpnet_davinci2", "Random"),
        ("oracle_mpnet_davinci2", "Oracle"),
        ("votek_mpnet_davinci2", "Vote-K"),
        ("ideal_mpnet_davinci2", "IDEAL"),
        ("quaild_comb_fl_mpnet_davinci2", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_davinci2", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_davinci2", "InSQuaD-LD"),
    )
    extra_column_tuples = (
        # gemma
        ("zeroshot_mpnet_gemma", "gemma2b"),
        ("random_mpnet_gemma", "gemma2b"),
        ("oracle_mpnet_gemma", "gemma2b"),
        ("votek_mpnet_gemma", "gemma2b"),
        ("ideal_mpnet_gemma", "gemma2b"),
        ("quaild_comb_fl_mpnet_gemma", "gemma2b"),
        ("quaild_comb_gc_mpnet_gemma", "gemma2b"),
        ("quaild_comb_ld_mpnet_gemma", "gemma2b"),
        # gemma7b
        ("hline", "hline"),
        ("zeroshot_mpnet_gemma7b", "gemma7b"),
        ("random_mpnet_gemma7b", "gemma7b"),
        ("oracle_mpnet_gemma7b", "gemma7b"),
        ("votek_mpnet_gemma7b", "gemma7b"),
        ("ideal_mpnet_gemma7b", "gemma7b"),
        ("quaild_comb_fl_mpnet_gemma7b", "gemma7b"),
        ("quaild_comb_gc_mpnet_gemma7b", "gemma7b"),
        ("quaild_comb_ld_mpnet_gemma7b", "gemma7b"),
        # davinci2
        ("hline", "hline"),
        ("zeroshot_mpnet_davinci2", "davinci2-175b"),
        ("random_mpnet_davinci2", "davinci2-175b"),
        ("oracle_mpnet_davinci2", "davinci2-175b"),
        ("votek_mpnet_davinci2", "davinci2-175b"),
        ("ideal_mpnet_davinci2", "davinci2-175b"),
        ("quaild_comb_fl_mpnet_davinci2", "davinci2-175b"),
        ("quaild_comb_gc_mpnet_davinci2", "davinci2-175b"),
        ("quaild_comb_ld_mpnet_davinci2", "davinci2-175b"),
    )
    extra_column_name = "Model"

    return generate_markdown_table(
        df, title, method_tuples, extra_column_name, extra_column_tuples
    )


def generate_main_table(df):
    """Generate main markdown table."""
    title = "Downstream evaluation on Gemma (2B)"
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

    return generate_markdown_table(df, title, method_tuples)


def generate_training_ablations_gemma(df):
    """Generate markdown table for training ablations."""
    title = "Downstream evaluation on Gemma (2B)"
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
        ("hline", "hline"),
        ("quaild_comb_fl_mpnet_gemma_best", "InSQuaD-FL"),
        ("quaild_comb_gc_mpnet_gemma_best", "InSQuaD-GC"),
        ("quaild_comb_ld_mpnet_gemma_best", "InSQuaD-LD"),
    )

    return generate_markdown_table(df, title, method_tuples)


if __name__ == "__main__":
    TABLES_TO_GENERATE = {
        "main_table": generate_main_table,
        "training_effect_gemmma": generate_training_ablations_gemma,
        "model_size_effect": generate_model_size_ablations,
        "qd_tradeoff_gemma": generate_qd_tradeoff_ablations_gemma,
        "annotation_budget_effect_gemma": generate_annotation_budget_ablations_gemma,
        "retrieval_method_effect_gemma": generate_retrieval_method_ablations_gemma,
    }
    BASE_PATH = Path("./artifacts/markdowns")
    BASE_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("./artifacts/tables/all.csv")

    for file_name, fn in tqdm(TABLES_TO_GENERATE.items()):
        with open(BASE_PATH / f"{file_name}.md", "w", encoding="utf-8") as f:
            f.write(fn(df))
