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


def prepare_table_dataframe(
    df,
    method_tuples,
    extra_column_name=None,
    extra_column_tuples=None,
    exclude_columns=None,
    column_order=None,
):
    """
    Prepare a DataFrame for table generation.

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    method_tuples (tuple): Method name tuples (id, display_name)
    extra_column_name (str, optional): Name for grouping column
    extra_column_tuples (tuple, optional): Extra column mapping tuples
    exclude_columns (list, optional): List of column names to exclude from the table

    Returns:
    pandas.DataFrame: Prepared DataFrame ready for markdown rendering
    dict: Dictionary mapping method names to dataset values
    list: List of dataset names in the data
    """
    import pandas as pd
    import numpy as np
    from scipy import stats

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

        # Store results with method name
        if extra_column_name and extra_column_tuples:
            # Use extra_column_lut to get the group name
            group_name = extra_column_lut.get(method_id, "")
            # Create combined key with group and method name
            combined_key = f"{group_name}: {method_name}"
            result_dict[combined_key] = method_data
        else:
            result_dict[method_name] = method_data

    # Convert result_dict to DataFrame for easier CSV export
    # Create a multi-index DataFrame from the result_dict
    if extra_column_name and extra_column_tuples:
        # Split the combined keys into group and method columns
        rows = []
        for combined_key, data in result_dict.items():
            group, method_name = combined_key.split(": ", 1)
            row_data = {"Group": group, "Method": method_name}
            row_data.update(data)
            rows.append(row_data)
        result_df = pd.DataFrame(rows)
    else:
        rows = []
        for method_name, data in result_dict.items():
            row_data = {"Method": method_name}
            row_data.update(data)
            rows.append(row_data)
        result_df = pd.DataFrame(rows)

    return result_df, result_dict, list(datasets)
