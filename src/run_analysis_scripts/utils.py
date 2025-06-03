import numpy as np
import pandas as pd
from scipy import stats
from dataloaders.dbpedia import DBPedia
from dataloaders.dummy import DummyDataset
from dataloaders.dummy_hotpot_qa_with_q_loader import DummyHotpotQaWithQDataset
from dataloaders.geoq import GeoQDataset
from dataloaders.hellaswag import Hellaswag
from dataloaders.hotpot_qa_loader import HotpotQaDataset
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset
from dataloaders.mnli import MNLI
from dataloaders.mrpc import MRPC
from dataloaders.mwoz import MwozDataset
from dataloaders.rte import RTE
from dataloaders.sst2 import SST2
from dataloaders.sst5 import SST5
from dataloaders.wiki_multihop_qa_loader import WikiMultihopQaDataset
from dataloaders.wiki_multihop_qa_with_q_loader import WikiMultihopQaWithQDataset
from dataloaders.xsum import XsumDataset

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
    MwozDataset.NAME: "MWoZ",
    GeoQDataset.NAME: "GeoQ",
}


GROUPS = {
    "Classification": [MRPC.NAME, SST5.NAME, MNLI.NAME, DBPedia.NAME, RTE.NAME],
    "Multi-Choice": [Hellaswag.NAME],
    "Dialogue": [MwozDataset.NAME],
    "Generation": [GeoQDataset.NAME, XsumDataset.NAME],
}


def dictify(tuple_of_tuples):
    result = {}
    for left, right in tuple_of_tuples:
        result[left] = right

    return result


def extract_relevant_df(full_df, method_tuples):
    full_df["method"] = full_df["method"].apply(lambda x: "_".join(x.split("_")[:-1]))

    relevant_methods = list(map(lambda x: x[0], method_tuples))
    df = full_df[full_df["method"].isin(relevant_methods)]
    return df


def generate_best_row(df):
    methods_fl = [
        "quaild_comb_fl_mpnet_gemma_lambda_0",
        "quaild_comb_fl_mpnet_gemma_lambda_025",
        "quaild_comb_fl_mpnet_gemma",
        "quaild_comb_fl_mpnet_gemma_lambda_1",
    ]
    methods_gc = [
        "quaild_comb_gc_mpnet_gemma_lambda_0",
        "quaild_comb_gc_mpnet_gemma_lambda_025",
        "quaild_comb_gc_mpnet_gemma",
        "quaild_comb_gc_mpnet_gemma_lambda_1",
    ]
    methods_ld = [
        "quaild_comb_ld_mpnet_gemma_lambda_0",
        "quaild_comb_ld_mpnet_gemma_lambda_025",
        "quaild_comb_ld_mpnet_gemma",
        "quaild_comb_ld_mpnet_gemma_lambda_1",
    ]

    ddf = df.copy()
    ddf["method"] = ddf["method"].apply(lambda x: "_".join(x.split("_")[:-1]))

    fl_max_values = ddf[ddf["method"].isin(methods_fl)].max()
    fl_max_values["method"] = "quaild_comb_fl_mpnet_gemma_best_00000"

    gc_max_values = ddf[ddf["method"].isin(methods_gc)].max()
    gc_max_values["method"] = "quaild_comb_gc_mpnet_gemma_best_00000"

    ld_max_values = ddf[ddf["method"].isin(methods_ld)].max()
    ld_max_values["method"] = "quaild_comb_ld_mpnet_gemma_best_00000"

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    fl_max_values,
                    gc_max_values,
                    ld_max_values,
                ]
            ),
        ],
        ignore_index=True,
    )

    return df


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


def prepare_table_dataframe(df, method_tuples, exclude_columns=None):
    # Initialize exclude_columns if None
    if exclude_columns is None:
        exclude_columns = ["seed"]

    df = extract_relevant_df(df.reset_index(), method_tuples)

    # Melt the DataFrame for easier manipulation
    df_melted = pd.melt(df, id_vars=["method"], var_name="dataset", value_name="value")

    # Exclude specified columns and default columns to exclude
    default_exclude = ["index", "Average", "seed"]
    all_exclude = list(set(default_exclude + exclude_columns))
    df_melted = df_melted[~df_melted["dataset"].isin(all_exclude)]

    # Get unique datasets
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
                    low, high = calculate_confidence_interval(values)
                    method_data[dataset] = f"{mean_val:.4f} ({low:.4f}, {high:.4f})"
                else:
                    method_data[dataset] = f"{mean_val:.4f} ERROR"
            else:
                method_data[dataset] = "-"

            result_dict[method_id] = method_data

    rows = []
    for method_name, data in result_dict.items():
        row_data = {"method": method_name}
        row_data.update(data)
        rows.append(row_data)
    result_df = pd.DataFrame(rows)

    return result_df
