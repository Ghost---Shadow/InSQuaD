import json
import unittest
from config import Config
from dataloaders.dummy import DummyDataset
from dense_indexes.wrapped_faiss import WrappedFaiss
from semantic_search_models.wrapped_mpnet import WrappedMpnetModel
from train_utils import set_seed


# python -m unittest dense_indexes.wrapped_faiss_test.TestFaissWrapper -v
class TestFaissWrapper(unittest.TestCase):
    # python -m unittest dense_indexes.wrapped_faiss_test.TestFaissWrapper.test_repopulate_index -v
    def test_repopulate_index(self):
        set_seed(42)

        config = Config.from_file("experiments/dummy_experiment.yaml")
        config.datasets.train = "dummy"
        config.architecture.dense_index.k_for_rerank = 2

        wrapped_dataset = DummyDataset(config)
        wrapped_mpnet_model = WrappedMpnetModel(config)
        wrapped_faiss = WrappedFaiss(config)

        wrapped_faiss.repopulate_index(wrapped_dataset, wrapped_mpnet_model)

        queries = wrapped_dataset.random_access([0, 1, 2])["prompts"]
        query_embeddings = wrapped_mpnet_model.embed(queries)
        retrieved_batch = wrapped_faiss.retrieve(query_embeddings)

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
                "distances": [0.5622506737709045, 0.5896707773208618],
            },
            {
                "prompts": [
                    "What is Charlie's favourite fruit?",
                    "What is Alice's favourite fruit?",
                ],
                "labels": ["coconut", "apple"],
                "distances": [0.5423349738121033, 0.5622507333755493],
            },
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Alice's favourite fruit?",
                ],
                "labels": ["banana", "apple"],
                "distances": [0.542335033416748, 0.5896708369255066],
            },
        ]

        # print(json.dumps(retrieved_batch))

        for actual, expected in zip(retrieved_batch, expected_batch):
            assert actual["prompts"] == expected["prompts"], json.dumps(actual)

        # Do it again
        wrapped_faiss.repopulate_index(wrapped_dataset, wrapped_mpnet_model)
        retrieved_batch = wrapped_faiss.retrieve(query_embeddings)
        assert retrieved_batch == expected_batch, json.dumps(retrieved_batch)
