import os
import unittest
from config import Config
from offline_eval_pipeline import OfflineEvaluationPipeline
from shortlist_strategies.vote_k import VoteKStrategy


# python -m unittest shortlist_strategies.vote_k_test.TestVoteK -v
class TestVoteK(unittest.TestCase):

    # python -m unittest shortlist_strategies.vote_k_test.TestVoteK.test_shortlist -v
    def test_shortlist(self):
        config = Config.from_file("experiments/tests/votek_test_experiment.yaml")
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"
        indexes, confidences = pipeline.shortlist_strategy.shortlist("mrpc")

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            1026,
            3489,
            666,
            1919,
            1973,
            2383,
            1831,
            2235,
            2303,
            3034,
            1848,
            2447,
            1538,
            1323,
            633,
            730,
            3630,
            326,
            777,
            818,
        ], indexes
        assert confidences == [
            0.9612076580524445,
            0.7525136321783066,
            0.5677379965782166,
            0.46843743324279785,
            0.30749231576919556,
            0.21573203802108765,
            0.20106947422027588,
            0.16396701335906982,
            0.11163413524627686,
            0.059060513973236084,
            0.03753089904785156,
            0.032401204109191895,
            0.015378236770629883,
            -0.0035791397094726562,
            -0.0070847272872924805,
            -0.007876038551330566,
            -0.03310680389404297,
            -0.05064702033996582,
            -0.05234181880950928,
            -0.05771017074584961,
        ], confidences

    # python -m unittest shortlist_strategies.vote_k_test.TestVoteK.test_drop_exact_duplicates -v
    def test_drop_exact_duplicates(self):
        batch_results = {
            "prompts": [
                "What is Bob's favourite fruit?",
                "What is Alice's favourite fruit?",
                "What is Alice's favourite fruit?",
                "What is Charlie's favourite fruit?",
                "What is Daphne's favourite fruit?",
            ],
            "labels": ["banana", "apple", "apple", "coconut", "durian"],
            "distances": [
                0.542335033416748,
                0.5896708965301514,
                0.5896708965301514,
                0.69,
                0.42,
            ],
            "global_indices": [1, 0, 0, 2, 3],
        }
        batch_query_prompts = [
            "What is Alice's favourite fruit?",
            "What is Daphne's favourite fruit?",
        ]
        actual = VoteKStrategy.drop_exact_duplicates(batch_results, batch_query_prompts)
        expected = {
            "prompts": [
                "What is Bob's favourite fruit?",
                "What is Charlie's favourite fruit?",
            ],
            "labels": ["banana", "coconut"],
            "distances": [0.542335033416748, 0.69],
            "global_indices": [
                1,
                2,
            ],
        }

        assert actual == expected, actual

    # python -m unittest shortlist_strategies.vote_k_test.TestVoteK.test_assemble_few_shot -v
    def test_assemble_few_shot(self):
        config = Config.from_file("experiments/tests/votek_test_experiment.yaml")
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
