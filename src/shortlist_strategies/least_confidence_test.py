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
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert indexes == [
            157,
            133,
            224,
            94,
            59,
            233,
            215,
            45,
            251,
            43,
            29,
            71,
            24,
            79,
            221,
            254,
            153,
            109,
            0,
            33,
        ], indexes
        assert confidences == [
            0.13465330004692078,
            0.13955064117908478,
            0.15235817432403564,
            0.1582164317369461,
            0.1603998839855194,
            0.16353042423725128,
            0.16524522006511688,
            0.17059928178787231,
            0.17839756608009338,
            0.19302403926849365,
            0.20361395180225372,
            0.20452596247196198,
            0.20828485488891602,
            0.20997728407382965,
            0.21962271630764008,
            0.24555368721485138,
            0.25548237562179565,
            0.2645929753780365,
            0.2755028009414673,
            0.2778550684452057,
        ], confidences

    # python -m unittest shortlist_strategies.least_confidence_test.TestLeastConfidenceStrategy.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file(
            "experiments/tests/leastconfidence_test_experiment.yaml"
        )
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        if not os.path.exists(pipeline.shortlisted_data_path):
            pipeline.shortlist()

        for row, few_shots in pipeline.shortlist_strategy.assemble_few_shot(
            "mrpc", use_cache=False
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
