import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
import torch
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

    # python -m unittest shortlist_strategies.insquad_mocombinatorial_test.TestInsquadMemoryOptimizedCombinatorialStrategy.test_shortlist_subsampled -v
    def test_shortlist_subsampled(self):
        config = Config.from_file("experiments/tests/insquad_motest_experiment.yaml")
        config.offline_validation.insquad_mocombinatorial_config.max_chunks_to_process = (
            90
        )
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
            1,
            93,
            166,
            271,
            181,
            95,
            95,
            18,
            98,
            79,
            163,
            106,
            130,
            96,
            3,
            170,
            138,
            77,
            251,
            116,
        ], indexes

        intersection = len(set(exact_indexes).intersection(set(indexes)))
        union = len(set(exact_indexes).union(set(indexes)))
        iou = intersection / union
        assert iou == 0.02631578947368421, iou

        assert confidences == [
            0.6081281900405884,
            -0.009695827960968018,
            -0.0033611655235290527,
            -0.0022423267364501953,
            -0.001494765281677246,
            -0.0013359785079956055,
            -0.0009542703628540039,
            -0.0011224150657653809,
            -0.0014519095420837402,
            -0.0013778209686279297,
            -0.0011748075485229492,
            -0.001124739646911621,
            -0.001026928424835205,
            -0.0010136961936950684,
            -0.0009434223175048828,
            -0.0008404254913330078,
            -0.0008604526519775391,
            -0.0008149147033691406,
            -0.0015192627906799316,
            -0.001993834972381592,
        ], confidences

    # python -m unittest shortlist_strategies.insquad_mocombinatorial_test.TestInsquadMemoryOptimizedCombinatorialStrategy.test_shortlist_subsampled_short -v
    def test_shortlist_subsampled_short(self):
        config = Config.from_file("experiments/tests/insquad_motest_experiment.yaml")
        config.offline_validation.insquad_mocombinatorial_config.max_chunks_to_process = (
            None
        )
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        # use_cache = False
        use_cache = True

        def mock_cache_similarities(use_cache):
            return torch.randn((1, 123, 123))

        pipeline.shortlist_strategy._cache_similarities = mock_cache_similarities
        indexes, confidences = pipeline.shortlist_strategy.shortlist(use_cache)

        assert indexes == [
            48,
            62,
            99,
            49,
            49,
            20,
            44,
            44,
            64,
            11,
            15,
            39,
            69,
            29,
            82,
            60,
            75,
            111,
            18,
            32,
        ], indexes

        assert confidences == [
            0.687820553779602,
            -0.19046634435653687,
            -0.07569175958633423,
            -0.04321613907814026,
            -0.025929629802703857,
            -0.02245357632637024,
            -0.01868826150894165,
            -0.014016181230545044,
            -0.015853703022003174,
            -0.018478453159332275,
            -0.0185040682554245,
            -0.019748836755752563,
            -0.018179520964622498,
            -0.016569241881370544,
            -0.01578652858734131,
            -0.01434837281703949,
            -0.013684973120689392,
            -0.015774980187416077,
            -0.014572970569133759,
            -0.033237844705581665,
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
