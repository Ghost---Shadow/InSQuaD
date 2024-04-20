import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest shortlist_strategies.least_confidence_test.TestLeastConfidenceStrategy -v
class TestLeastConfidenceStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.least_confidence_test.TestLeastConfidenceStrategy.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file(
            "experiments/tests/leastconfidence_test_experiment.yaml"
        )
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        indexes, confidences = pipeline.shortlist_strategy.shortlist()

        assert indexes == [
            628,
            842,
            1160,
            2335,
            1457,
            1556,
            2667,
            1367,
            287,
            2978,
            1514,
            2411,
            1977,
            882,
            2177,
            1580,
            418,
            2165,
            1355,
            710,
        ], indexes
        assert confidences == [
            0.03724600747227669,
            0.04732665419578552,
            0.05275539681315422,
            0.052786558866500854,
            0.055073924362659454,
            0.05930019170045853,
            0.06320629268884659,
            0.07422198355197906,
            0.07641955465078354,
            0.07648376375436783,
            0.0848764181137085,
            0.08602114021778107,
            0.08661555498838425,
            0.08821915090084076,
            0.0884052962064743,
            0.08892496675252914,
            0.0933811366558075,
            0.09409453719854355,
            0.10303119570016861,
            0.11095193773508072,
        ], confidences

    # python -m unittest shortlist_strategies.least_confidence_test.TestLeastConfidenceStrategy.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file(
            "experiments/tests/leastconfidence_test_experiment.yaml"
        )
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        pipeline.shortlist()

        for row, few_shots in pipeline.shortlist_strategy.assemble_few_shot(
            use_cache=False
        ):
            assert "prompts" in row, row
            assert "labels" in row, row

            assert "prompts" in few_shots, few_shots
            assert "labels" in few_shots, few_shots

            assert (
                len(few_shots["prompts"]) == config.offline_validation.num_shots
            ), few_shots
            assert (
                len(few_shots["labels"]) == config.offline_validation.num_shots
            ), few_shots
