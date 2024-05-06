import inspect
import os
import unittest
import pandas as pd
import numpy as np
from run_analysis_scripts.excelify import excelify
from run_analysis_scripts.tables_in_paper import (
    DATASET_NAME_KEYS,
    GROUPS,
    generate_annotation_budget_ablations,
    generate_latex_table,
    generate_qd_tradeoff_ablations,
    generate_retrieval_method_ablations,
    get_column_spec,
)

SNAPSHOT_PATH = "src/run_analysis_scripts/snapshots/"


def get_invoking_function_name():
    caller_frame = inspect.stack()[2]
    caller_function_name = caller_frame.function
    return caller_function_name


def snapshot_helper(result, update=False):
    name = get_invoking_function_name()
    path = f"{SNAPSHOT_PATH}/{name}.txt"
    if (not os.path.exists(path)) or update:
        with open(path, "w") as f:
            f.write(result)

    with open(path) as f:
        snapshot = f.read()

    assert result == snapshot


# python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable -v
class TestGenerateLatexTable(unittest.TestCase):
    def setUp(self):
        data = {
            "method": [
                "zeroshot_test_experiment_b6952",
                "leastconfidence_test_experiment_ae3b7",
                "random_test_experiment_ae3b7",
            ],
            "mrpc": [88.5, np.nan, 23.5],
            "sst5": [45.6, 43.2, 32.5],
            "mnli": [75.0, np.nan, 34.3],
            "dbpedia": [np.nan, 95.0, 22.3],
            "rte": [np.nan, 84.1, 32.3],
            "hellaswag": [60.2, 61.5, 34.4],
            "mwoz": [np.nan, np.nan, np.nan],
            "geoq": [70.1, 70.2, np.nan],
            "xsum": [np.nan, 48.3, np.nan],
        }
        self.df = pd.DataFrame(data)

    # python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable.test_generate_latex_table -v
    def test_generate_latex_table(self):
        caption = "Results on gemma"
        label = "gemma_results"
        method_lut = {
            "zeroshot_test_experiment": "Zeroshot",
            "random_test_experiment": "Random",
        }
        result = generate_latex_table(self.df, caption, label, method_lut)
        snapshot_helper(result, update=True)

    # python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable.test_generate_latex_table_real -v
    def test_generate_latex_table_real(self):
        df = excelify()
        caption = "Results on gemma"
        label = "gemma_results"
        method_lut = {
            "zeroshot_test_experiment": "Zeroshot",
            "random_test_experiment": "Random",
        }
        result = generate_latex_table(df, caption, label, method_lut)
        snapshot_helper(result, update=True)

    # python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable.test_generate_retrieval_method_ablations -v
    def test_generate_retrieval_method_ablations(self):
        df = excelify()
        result = generate_retrieval_method_ablations(df)
        snapshot_helper(result, update=True)

    # python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable.test_generate_annotation_budget_ablations -v
    def test_generate_annotation_budget_ablations(self):
        df = excelify()
        result = generate_annotation_budget_ablations(df)
        snapshot_helper(result, update=True)

    # python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable.test_generate_qd_tradeoff_ablations -v
    def test_generate_qd_tradeoff_ablations(self):
        df = excelify()
        result = generate_qd_tradeoff_ablations(df)
        snapshot_helper(result, update=True)


# python -m unittest run_analysis_scripts.tables_in_paper_test.TestLaTeXColumnSpec -v
class TestLaTeXColumnSpec(unittest.TestCase):

    def test_column_spec_and_multicolumn_with_dict(self):
        expected_column_spec = "l|ccccc|c|c|cc"
        expected_multicolumn_line = "\\textbf{Method} & \\multicolumn{5}{c|}{\\textbf{Classification}} & \\multicolumn{1}{c|}{\\textbf{Multi-Choice}} & \\multicolumn{1}{c|}{\\textbf{Dialogue}} & \\multicolumn{2}{c}{\\textbf{Generation}}"

        column_spec, multicolumn_line = get_column_spec(GROUPS, DATASET_NAME_KEYS)
        self.maxDiff = None
        self.assertEqual(column_spec, expected_column_spec)
        self.assertEqual(multicolumn_line, expected_multicolumn_line)


if __name__ == "__main__":
    unittest.main()
