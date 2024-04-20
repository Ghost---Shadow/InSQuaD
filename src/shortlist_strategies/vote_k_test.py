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
        indexes, confidences = pipeline.shortlist_strategy.shortlist()

        assert len(indexes) == config.offline_validation.annotation_budget, len(indexes)

        assert indexes == [
            3,
            165,
            43,
            36,
            42,
            33,
            175,
            59,
            61,
            204,
            28,
            30,
            17,
            279,
            23,
            84,
            226,
            77,
            105,
            104,
        ], indexes
        assert confidences == [
            0.0061331987380981445,
            -0.10036945343017578,
            -0.10135483741760254,
            -0.10500609874725342,
            -0.11835360527038574,
            -0.14868950843811035,
            -0.1513378620147705,
            -0.1575937271118164,
            -0.16222691535949707,
            -0.16232848167419434,
            -0.18247485160827637,
            -0.189894437789917,
            -0.19379210472106934,
            -0.21715831756591797,
            -0.22450506687164307,
            -0.22792792320251465,
            -0.23586344718933105,
            -0.248612642288208,
            -0.2530827522277832,
            -0.26886487007141113,
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

    # python -m unittest shortlist_strategies.vote_k_test.TestVoteK.test_oom -v
    def test_oom(self):
        config = Config.from_file(
            "experiments/tests/shortlistandtopk_test_experiment.yaml"
        )
        config.offline_validation.subsample_for_train_size = 3000
        pipeline = OfflineEvaluationPipeline(config)
        pipeline.set_seed(42)
        pipeline.current_dataset_name = "mrpc"

        # Should not OOM
        pipeline.shortlist_strategy.shortlist()
