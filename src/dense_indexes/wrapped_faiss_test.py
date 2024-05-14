import json
import unittest
from config import Config
from dataloaders.dummy import DummyDataset
from dense_indexes.wrapped_faiss import WrappedFaiss
from train_utils import set_seed
from training_pipeline import TrainingPipeline


# python -m unittest dense_indexes.wrapped_faiss_test.TestFaissWrapper -v
class TestFaissWrapper(unittest.TestCase):
    # python -m unittest dense_indexes.wrapped_faiss_test.TestFaissWrapper.test_repopulate_index -v
    def test_repopulate_index(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.dense_index.k_for_rerank = 2
        pipeline = TrainingPipeline(config)
        pipeline.set_seed(42)

        wrapped_dataset = DummyDataset(config)
        wrapped_mpnet_model = pipeline.semantic_search_model
        wrapped_faiss = WrappedFaiss(config, pipeline)

        wrapped_faiss.repopulate_index(wrapped_dataset, wrapped_mpnet_model)

        queries = wrapped_dataset.random_access([0, 1, 2])["prompts"]
        query_embeddings = wrapped_mpnet_model.embed(queries)
        retrieved_batch = wrapped_faiss.retrieve(query_embeddings, omit_self=True)

        # Sanity check
        assert queries == [
            "What is Alice's favourite fruit?",
            "What is Bob's favourite fruit?",
            "What is Charlie's favourite fruit?",
        ], queries

        expected_batch = [
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Charlie's favourite fruit?",
                ],
                "labels": ["banana", "coconut"],
                "distances": [0.5622506141662598, 0.5896706581115723],
                "global_indices": [1, 2],
            },
            {
                "prompts": [
                    "What is Charlie's favourite fruit?",
                    "What is Alice's favourite fruit?",
                ],
                "labels": ["coconut", "apple"],
                "distances": [0.542334794998169, 0.5622506141662598],
                "global_indices": [2, 0],
            },
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Alice's favourite fruit?",
                ],
                "labels": ["banana", "apple"],
                "distances": [0.5423349738121033, 0.5896711349487305],
                "global_indices": [1, 0],
            },
        ]
        # print(json.dumps(retrieved_batch))

        for actual, expected in zip(retrieved_batch, expected_batch):
            assert actual["prompts"] == expected["prompts"], json.dumps(actual)

        # Do it again
        wrapped_faiss.repopulate_index(wrapped_dataset, wrapped_mpnet_model)
        retrieved_batch = wrapped_faiss.retrieve(query_embeddings, omit_self=True)
        assert retrieved_batch == expected_batch, json.dumps(retrieved_batch)

    # python -m unittest dense_indexes.wrapped_faiss_test.TestFaissWrapper.test_repopulate_index_no_omit -v
    def test_repopulate_index_no_omit(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.architecture.dense_index.k_for_rerank = 2
        pipeline = TrainingPipeline(config)
        pipeline.set_seed(42)

        wrapped_dataset = DummyDataset(config)
        wrapped_mpnet_model = pipeline.semantic_search_model
        wrapped_faiss = WrappedFaiss(config, pipeline)

        wrapped_faiss.repopulate_index(wrapped_dataset, wrapped_mpnet_model)

        queries = wrapped_dataset.random_access([0, 1, 2])["prompts"]
        query_embeddings = wrapped_mpnet_model.embed(queries)
        retrieved_batch = wrapped_faiss.retrieve(query_embeddings, omit_self=False)

        # Sanity check
        assert queries == [
            "What is Alice's favourite fruit?",
            "What is Bob's favourite fruit?",
            "What is Charlie's favourite fruit?",
        ], queries

        expected_batch = [
            {
                "prompts": [
                    "What is Alice's favourite fruit?",
                    "What is Bob's favourite fruit?",
                ],
                "labels": ["apple", "banana"],
                "distances": [3.414730520914788e-13, 0.5622506141662598],
                "global_indices": [0, 1],
            },
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Charlie's favourite fruit?",
                ],
                "labels": ["banana", "coconut"],
                "distances": [2.667220285780536e-13, 0.542334794998169],
                "global_indices": [1, 2],
            },
            {
                "prompts": [
                    "What is Charlie's favourite fruit?",
                    "What is Bob's favourite fruit?",
                ],
                "labels": ["coconut", "banana"],
                "distances": [4.06841606844649e-13, 0.5423349738121033],
                "global_indices": [2, 1],
            },
        ]

        # print(json.dumps(retrieved_batch))

        for actual, expected in zip(retrieved_batch, expected_batch):
            assert actual["prompts"] == expected["prompts"], json.dumps(actual)

        # Do it again
        wrapped_faiss.repopulate_index(wrapped_dataset, wrapped_mpnet_model)
        retrieved_batch = wrapped_faiss.retrieve(query_embeddings, omit_self=False)
        assert retrieved_batch == expected_batch, json.dumps(retrieved_batch)
