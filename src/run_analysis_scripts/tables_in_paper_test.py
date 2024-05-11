import inspect
import os
import unittest
import pandas as pd
import numpy as np
from run_analysis_scripts.excelify import excelify
from run_analysis_scripts.tables_in_paper import (
    GROUPS,
    generate_latex_rows,
    generate_latex_table,
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
    print(path)
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
            "mrpc": [0.885, np.nan, 0.235],
            "sst5": [0.456, 0.432, 0.325],
            "mnli": [0.750, np.nan, 0.343],
            "dbpedia": [np.nan, 0.950, 0.223],
            "rte": [np.nan, 0.841, 0.323],
            "hellaswag": [0.602, 0.615, 0.344],
            "mwoz": [np.nan, np.nan, np.nan],
            "geoq": [0.701, 0.702, np.nan],
            "xsum": [np.nan, 0.483, np.nan],
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

    # python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable.test_column_spec_and_multicolumn_with_dict -v
    def test_column_spec_and_multicolumn_with_dict(self):
        expected_column_spec = "l|l|ccccc|c|c|cc"
        expected_multicolumn_line = "\\textbf{Method} & \\textbf{POTATO} & \\multicolumn{5}{c|}{\\textbf{Classification}} & \\multicolumn{1}{c|}{\\textbf{Multi-Choice}} & \\multicolumn{1}{c|}{\\textbf{Dialogue}} & \\multicolumn{2}{c}{\\textbf{Generation}}"

        column_spec, multicolumn_line = get_column_spec(
            GROUPS, extra_column_name="POTATO"
        )
        self.maxDiff = None
        self.assertEqual(column_spec, expected_column_spec)
        self.assertEqual(multicolumn_line, expected_multicolumn_line)

    # python -m unittest run_analysis_scripts.tables_in_paper_test.TestGenerateLatexTable.test_generate_latex_rows -v
    def test_generate_latex_rows(self):
        data = {
            "method": [
                "random_mpnet_stablelm_c00de",
                "random_mpnet_stablelm_100_175f7",
            ],
            "mrpc": [1, 2],
            "sst5": [1, 2],
            "mnli": [1, 2],
            "dbpedia": [1, 2],
            "rte": [1, 2],
            "hellaswag": [1, 2],
            "mwoz": [1, 2],
            "geoq": [1, 2],
            "xsum": [1, 2],
        }
        df = pd.DataFrame(data)
        method_lut = {
            "random_mpnet_stablelm": "Random",
            "random_mpnet_stablelm_100": "Random 100",
        }
        expected_columns = [col for group in GROUPS.values() for col in group]
        num_columns = len(expected_columns)
        extra_column_lut = None
        rows = generate_latex_rows(df, method_lut, num_columns, extra_column_lut)

        expected_line_1 = "Random & 100.0 & 100.0 & 100.0 & 100.0 & 100.0 & 100.0 & 100.0 & 100.0 & 100.0 \\\\"
        expected_line_2 = "Random 100 & 200.0 & 200.0 & 200.0 & 200.0 & 200.0 & 200.0 & 200.0 & 200.0 & 200.0 \\\\"

        line_1, line_2, line_3 = rows.split("\n")

        self.assertEqual(line_1, expected_line_1)
        self.assertEqual(line_2, expected_line_2)
        self.assertEqual(line_3, "")


if __name__ == "__main__":
    unittest.main()
