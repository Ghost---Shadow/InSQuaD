from dataloaders.dbpedia import DBPedia
from dataloaders.dummy import DummyDataset
from dataloaders.dummy_hotpot_qa_with_q_loader import DummyHotpotQaWithQDataset
from dataloaders.hellaswag import Hellaswag
from dataloaders.hotpot_qa_loader import HotpotQaDataset
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset
from dataloaders.mnli import MNLI
from dataloaders.mrpc import MRPC
from dataloaders.rte import RTE
from dataloaders.sst2 import SST2
from dataloaders.sst5 import SST5
from dataloaders.wiki_multihop_qa_loader import WikiMultihopQaDataset
from dataloaders.wiki_multihop_qa_with_q_loader import WikiMultihopQaWithQDataset
from dataloaders.xsum import XsumDataset
import pandas as pd
import numpy as np

DATASET_NAME_KEYS = {
    DBPedia.NAME: "DBpedia",
    DummyDataset.NAME: "DummyDataset",
    DummyHotpotQaWithQDataset.NAME: "DummyHotpotQaWithQDataset",
    Hellaswag.NAME: "HellaSwag",
    HotpotQaDataset.NAME: "HotpotQaDataset",
    HotpotQaWithQDataset.NAME: "HotpotQaWithQDataset",
    MNLI.NAME: "MNLI",
    MRPC.NAME: "MRPC",
    RTE.NAME: "RTE",
    SST2.NAME: "SST2",
    SST5.NAME: "SST5",
    WikiMultihopQaDataset.NAME: "WikiMultihopQaDataset",
    WikiMultihopQaWithQDataset.NAME: "WikiMultihopQaWithQDataset",
    XsumDataset.NAME: "Xsum",
    "mwoz": "MWoZ",  # TODO
    "geoq": "GeoQ",  # TODO
}


GROUPS = {
    "Classification": [MRPC.NAME, SST5.NAME, MNLI.NAME, DBPedia.NAME, RTE.NAME],
    "Multi-Choice": [Hellaswag.NAME],
    "Dialogue": ["mwoz"],
    "Generation": ["geoq", XsumDataset.NAME],
}


def get_column_spec(groups, extra_column_name):
    column_spec = "l"  # Start with 'l' for the 'Method' column
    if extra_column_name:
        column_spec += "|l"
    multicolumn_parts = []
    total_columns = sum(len(columns) for columns in groups.values())

    current_column = 1  # Starting after 'Method' column which is 1

    for group, columns in groups.items():
        column_count = len(columns)
        column_spec += "|" + "c" * column_count

        # Decide if a vertical bar should be added at the end of this multicolumn
        current_column += column_count
        if current_column < total_columns:
            multicolumn_parts.append(
                f"\\multicolumn{{{column_count}}}{{c|}}{{\\textbf{{{group}}}}}"
            )
        else:
            multicolumn_parts.append(
                f"\\multicolumn{{{column_count}}}{{c}}{{\\textbf{{{group}}}}}"
            )

    multicolumn_line = " & ".join(multicolumn_parts)
    if extra_column_name:
        multicolumn_line = f"\\textbf{{{extra_column_name}}} & " + multicolumn_line

    multicolumn_line = "\\textbf{Method} & " + multicolumn_line
    return column_spec, multicolumn_line


def generate_latex_rows(df, method_lut, num_columns, extra_column_lut):
    latex_rows = ""
    for method, method_print_name in method_lut.items():
        row = df[df["method"].str.startswith(method)]
        method_column = [method_print_name]
        extra_column = []
        if extra_column_lut is not None:
            extra_column = [extra_column_lut[method]]
        if len(row) == 1:
            row = row.values.tolist()[0]
            cells = (
                method_column
                + extra_column
                + [f"{x:.1f}" if pd.notna(x) else "??.?" for x in row[1:]]
            )
        else:
            cells = method_column + extra_column + ["??.?"] * num_columns
        latex_row = " & ".join(cells)
        latex_rows += latex_row + " \\\\\n"
    return latex_rows


def generate_latex_table(
    df, caption, label, method_lut, extra_column_name=None, extra_column_lut=None
):
    # Reset the index to make 'method' a regular column
    df = df.reset_index()

    if extra_column_lut is not None:
        assert set(extra_column_lut.keys()) == set(method_lut.keys())
        assert extra_column_name is not None

    # Ensure all required columns are present in the DataFrame
    expected_columns = [col for group in GROUPS.values() for col in group]
    column_order = ["method"] + expected_columns
    for column in expected_columns:
        if column.lower() not in df.columns.str.lower():
            df[column] = np.nan  # Add missing columns as NaN

    # Reorder DataFrame to match the exact column order needed
    df = df[column_order]

    num_columns = len(expected_columns)
    latex_rows = generate_latex_rows(df, method_lut, num_columns, extra_column_lut)
    column_spec, multicolumn_line = get_column_spec(GROUPS, extra_column_name)

    header_line = "& " + " & ".join(
        [
            f"\\textbf{{{DATASET_NAME_KEYS[col]}}}"
            for group in GROUPS.values()
            for col in group
        ]
    )

    if extra_column_name is not None:
        header_line = "&" + header_line

    offset = 0
    if extra_column_name is not None:
        offset = 1

    latex_template = f"""
\\begin{{table}}[H]
\\centering
\\small
\\caption{{{caption}}}
\\label{{table:{label}}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{{column_spec}}}
\\hline
{multicolumn_line} \\\\
\\cmidrule(lr){{{2+offset}-{len(column_order)+offset}}}
{header_line} \\\\
\\hline
{latex_rows}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return latex_template


def generate_retrieval_method_ablations(df):
    caption = "Effect of retrieval methods StableLM (1.6B)"
    label = "retrieval_method_ablations"
    method_lut = {
        "zeroshot_mpnet_stablelm": "Zeroshot",
        "random_mpnet_stablelm": "Random",
        "quaild_random_fl_mpnet_stablelm": "QuailD-FL",
        "quaild_random_gc_mpnet_stablelm": "QuailD-GC",
        "quaild_similar_fl_mpnet_stablelm": "QuailD-FL",
        "quaild_similar_gc_mpnet_stablelm": "QuailD-GC",
        "quaild_gain_fl_mpnet_stablelm": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm": "QuailD-GC",
    }
    extra_column_lut = {
        "zeroshot_mpnet_stablelm": "",
        "random_mpnet_stablelm": "",
        "quaild_random_fl_mpnet_stablelm": "Random",
        "quaild_random_gc_mpnet_stablelm": "Random",
        "quaild_similar_fl_mpnet_stablelm": "Similar",
        "quaild_similar_gc_mpnet_stablelm": "Similar",
        "quaild_gain_fl_mpnet_stablelm": "Submodular",
        "quaild_gain_gc_mpnet_stablelm": "Submodular",
    }
    extra_column_name = "Retrieval"
    result = generate_latex_table(
        df,
        caption,
        label,
        method_lut,
        extra_column_name,
        extra_column_lut,
    )

    return result


def generate_annotation_budget_ablations(df):
    caption = "Effects of annotation budget StableLM (1.6B)"
    label = "budget_ablations"
    method_lut = {
        "zeroshot_mpnet_stablelm": "Zeroshot",
        # budget 18
        "random_mpnet_stablelm": "Random",
        "vote_k_stablelm": "Vote-K",
        "ideal_stablelm": "IDEAL",
        "quaild_gain_fl_mpnet_stablelm": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm": "QuailD-GC",
        # budget 100
        "random_mpnet_stablelm_100": "Random",
        "vote_k_stablelm_100": "Vote-K",
        "ideal_stablelm_100": "IDEAL",
        "quaild_gain_fl_mpnet_stablelm_100": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm_100": "QuailD-GC",
    }
    extra_column_lut = {
        # Zeroshot
        "zeroshot_mpnet_stablelm": "",
        # budget 18
        "random_mpnet_stablelm": "18",
        "vote_k_stablelm": "18",
        "ideal_stablelm": "18",
        "quaild_gain_fl_mpnet_stablelm": "18",
        "quaild_gain_gc_mpnet_stablelm": "18",
        # budget 100
        "random_mpnet_stablelm_100": "100",
        "vote_k_stablelm_100": "100",
        "ideal_stablelm_100": "100",
        "quaild_gain_fl_mpnet_stablelm_100": "100",
        "quaild_gain_gc_mpnet_stablelm_100": "100",
    }
    extra_column_name = "Budget"
    result = generate_latex_table(
        df,
        caption,
        label,
        method_lut,
        extra_column_name,
        extra_column_lut,
    )

    return result


def generate_qd_tradeoff_ablations(df):
    caption = "Effects of $\\lambda$ on StableLM (1.6B) (Quality-Diversity tradeoff)"
    label = "qd_tradeoff"
    method_lut = {
        "zeroshot_mpnet_stablelm": "Zeroshot",
        "random_mpnet_stablelm": "Random",
        "quaild_gain_fl_mpnet_stablelm_lambda_0": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm_lambda_0": "QuailD-GC",
        "quaild_gain_fl_mpnet_stablelm": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm": "QuailD-GC",
        "quaild_gain_fl_mpnet_stablelm_lambda_1": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm_lambda_1": "QuailD-GC",
    }
    extra_column_lut = {
        "zeroshot_mpnet_stablelm": "",
        "random_mpnet_stablelm": "",
        "quaild_gain_fl_mpnet_stablelm_lambda_0": "0",
        "quaild_gain_gc_mpnet_stablelm_lambda_0": "0",
        "quaild_gain_fl_mpnet_stablelm": "0.5",
        "quaild_gain_gc_mpnet_stablelm": "0.5",
        "quaild_gain_fl_mpnet_stablelm_lambda_1": "1",
        "quaild_gain_gc_mpnet_stablelm_lambda_1": "1",
    }
    extra_column_name = "$\\lambda$"
    result = generate_latex_table(
        df,
        caption,
        label,
        method_lut,
        extra_column_name,
        extra_column_lut,
    )

    return result


def generate_model_size_ablations(df):
    caption = "Effects of model size"
    label = "model_size"
    method_lut = {
        # stablelm
        "zeroshot_mpnet_stablelm": "Zeroshot",
        "random_mpnet_stablelm": "Random",
        "vote_k_stablelm": "Vote-K",
        "ideal_stablelm": "IDEAL",
        "quaild_gain_fl_mpnet_stablelm": "QuailD-FL",
        "quaild_gain_gc_mpnet_stablelm": "QuailD-GC",
        # llama7b
        "zeroshot_mpnet_llama7b": "Zeroshot",
        "random_mpnet_llama7b": "Random",
        "vote_k_llama7b": "Vote-K",
        "ideal_llama7b": "IDEAL",
        "quaild_gain_fl_mpnet_llama7b": "QuailD-FL",
        "quaild_gain_gc_mpnet_llama7b": "QuailD-GC",
        # davinci2
        "zeroshot_mpnet_davinci2": "Zeroshot",
        "random_mpnet_davinci2": "Random",
        "vote_k_davinci2": "Vote-K",
        "ideal_davinci2": "IDEAL",
        "quaild_gain_fl_mpnet_davinci2": "QuailD-FL",
        "quaild_gain_gc_mpnet_davinci2": "QuailD-GC",
    }
    extra_column_lut = {
        # stablelm
        "zeroshot_mpnet_stablelm": "StableLM1.6B",
        "random_mpnet_stablelm": "StableLM1.6B",
        "vote_k_stablelm": "StableLM1.6B",
        "ideal_stablelm": "StableLM1.6B",
        "quaild_gain_fl_mpnet_stablelm": "StableLM1.6B",
        "quaild_gain_gc_mpnet_stablelm": "StableLM1.6B",
        # llama7b
        "zeroshot_mpnet_llama7b": "llama7b",
        "random_mpnet_llama7b": "llama7b",
        "vote_k_llama7b": "llama7b",
        "ideal_llama7b": "llama7b",
        "quaild_gain_fl_mpnet_llama7b": "llama7b",
        "quaild_gain_gc_mpnet_llama7b": "llama7b",
        # davinci2
        "zeroshot_mpnet_davinci2": "davinci2-175b",
        "random_mpnet_davinci2": "davinci2-175b",
        "vote_k_davinci2": "davinci2-175b",
        "ideal_davinci2": "davinci2-175b",
        "quaild_gain_fl_mpnet_davinci2": "davinci2-175b",
        "quaild_gain_gc_mpnet_davinci2": "davinci2-175b",
    }
    extra_column_name = "Model"
    result = generate_latex_table(
        df,
        caption,
        label,
        method_lut,
        extra_column_name,
        extra_column_lut,
    )

    return result
