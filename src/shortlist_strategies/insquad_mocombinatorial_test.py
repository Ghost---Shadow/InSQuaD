import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
import yaml


# python -m unittest shortlist_strategies.insquad_mocombinatorial_test.TestInsquadMemoryOptimizedCombinatorialStrategy -v
class TestInsquadMemoryOptimizedCombinatorialStrategy(unittest.TestCase):
    # python -m unittest shortlist_strategies.insquad_mocombinatorial_test.TestInsquadMemoryOptimizedCombinatorialStrategy.test_config -v
    def test_config(self):
        with open("experiments/tests/insquad_motest_experiment.yaml", "r") as file:
            config_data = yaml.safe_load(file)

        # Should not crash
        Config.from_dict(config_data)

        config_data["offline_validation"]["insquad_mocombinatorial_config"][
            "window_size"
        ] = 41

        with self.assertRaises(ValueError) as e:
            config = Config.from_dict(config_data)
            OfflineEvaluationPipeline(config)

        self.assertEqual(
            e.exception.args[0],
            "offline_validation.insquad_mocombinatorial_config.window_size is not divisible by offline_validation.subsample_for_train_size",
            str(e.exception.args[0]),
        )

    # python -m unittest shortlist_strategies.insquad_mocombinatorial_test.TestInsquadMemoryOptimizedCombinatorialStrategy.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/tests/insquad_motest_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        # use_cache = False
        use_cache = True
        indexes, confidences = pipeline.shortlist_strategy.shortlist(use_cache)

        exact_indexes = [
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
        ]

        assert indexes == [
            179,
            91,
            190,
            1,
            13,
            289,
            148,
            155,
            226,
            123,
            248,
            76,
            166,
            66,
            272,
            111,
            98,
            100,
            260,
            11,
        ], indexes

        intersection = len(set(exact_indexes).intersection(set(indexes)))
        union = len(set(exact_indexes).union(set(indexes)))
        iou = intersection / union
        assert iou == 0.3333333333333333, iou

        # Exact confidences
        # assert confidences == [
        #     0.5913819670677185,
        #     -0.0022861361503601074,
        #     -0.0009650588035583496,
        #     -0.0013608932495117188,
        #     -0.0009065866470336914,
        #     -0.0008512139320373535,
        #     -0.0006109476089477539,
        #     -0.0006862878799438477,
        #     -0.0005466938018798828,
        #     -0.0004929900169372559,
        #     -0.00048160552978515625,
        #     -0.00040262937545776367,
        #     -0.00037169456481933594,
        #     -0.00032651424407958984,
        #     -0.00036263465881347656,
        #     -0.0003325343132019043,
        #     -0.0003428459167480469,
        #     -0.0003165602684020996,
        #     -0.00028389692306518555,
        #     -0.00026601552963256836,
        # ], confidences
        assert confidences == [
            0.6415250897407532,
            -0.0027306079864501953,
            -0.0023728013038635254,
            -0.0012278556823730469,
            -0.0008025765419006348,
            -0.0008150339126586914,
            -0.0008904933929443359,
            -0.0007295608520507812,
            -0.001280665397644043,
            -0.0010495185852050781,
            -0.0008848309516906738,
            -0.0009338855743408203,
            -0.0009608268737792969,
            -0.0014004111289978027,
            -0.0013683438301086426,
            -0.0015504956245422363,
            -0.0017457008361816406,
            -0.0016956329345703125,
            -0.0016676783561706543,
            -0.0015034079551696777,
        ], confidences

    # python -m unittest shortlist_strategies.insquad_mocombinatorial_test.TestInsquadMemoryOptimizedCombinatorialStrategy.test_assemble_few_shot -v
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
