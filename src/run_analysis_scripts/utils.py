import pandas as pd


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
        "quaild_gain_fl_mpnet_gemma_lambda_0",
        "quaild_gain_fl_mpnet_gemma_lambda_025",
        "quaild_gain_fl_mpnet_gemma",
        "quaild_gain_fl_mpnet_gemma_lambda_1",
    ]
    methods_gc = [
        "quaild_gain_gc_mpnet_gemma_lambda_0",
        "quaild_gain_gc_mpnet_gemma_lambda_025",
        "quaild_gain_gc_mpnet_gemma",
        "quaild_gain_gc_mpnet_gemma_lambda_1",
    ]

    ddf = df.copy()
    ddf["method"] = ddf["method"].apply(lambda x: "_".join(x.split("_")[:-1]))

    fl_max_values = ddf[ddf["method"].isin(methods_fl)].max()
    fl_max_values["method"] = "quaild_gain_fl_mpnet_gemma_best_00000"

    gc_max_values = ddf[ddf["method"].isin(methods_gc)].max()
    gc_max_values["method"] = "quaild_gain_gc_mpnet_gemma_best_00000"

    df = pd.concat(
        [df, pd.DataFrame([fl_max_values, gc_max_values])], ignore_index=True
    )

    return df
