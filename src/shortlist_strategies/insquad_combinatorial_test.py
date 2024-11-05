import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline


# python -m unittest shortlist_strategies.insquad_combinatorial_test.TestInsquadCombinatorialStrategy -v
class TestInsquadCombinatorialStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.insquad_combinatorial_test.TestInsquadCombinatorialStrategy.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/tests/insquad_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        use_cache = False
        # use_cache = True
        indexes, confidences = pipeline.shortlist_strategy.shortlist(use_cache)

        assert indexes == [
            1,
            179,
            127,
            13,
            155,
            91,
            190,
            148,
            47,
            16,
            123,
            289,
            4,
            248,
            132,
            255,
            220,
            193,
            131,
            150,
        ], indexes
        assert confidences == [
            0.5913819670677185,
            -0.0022861361503601074,
            -0.0009650588035583496,
            -0.0013608932495117188,
            -0.0009065866470336914,
            -0.0008512139320373535,
            -0.0006109476089477539,
            -0.0006862878799438477,
            -0.0005466938018798828,
            -0.0004929900169372559,
            -0.00048160552978515625,
            -0.00040262937545776367,
            -0.00037169456481933594,
            -0.00032651424407958984,
            -0.00036263465881347656,
            -0.0003325343132019043,
            -0.0003428459167480469,
            -0.0003165602684020996,
            -0.00028389692306518555,
            -0.00026601552963256836,
        ], confidences

    # python -m unittest shortlist_strategies.insquad_combinatorial_test.TestInsquadCombinatorialStrategy.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/insquad_test_experiment.yaml")
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
