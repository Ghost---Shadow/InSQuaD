import unittest
import torch
import torch.optim as optim
import numpy as np
from semantic_search_models import WrappedMpnetModel
from src.config import Config


# python -m unittest semantic_search_models.wrapped_mpnet_test.TestWrappedMpnetModel -v
class TestWrappedMpnetModel(unittest.TestCase):

    # python -m unittest semantic_search_models.wrapped_mpnet_test.TestWrappedMpnetModel.test_embed -v
    def test_embed(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        model = WrappedMpnetModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        # Embed the query and the documents
        all_embeddings = model.embed([query, *documents])
        query_embedding = all_embeddings[0]
        document_embeddings = all_embeddings[1:]

        # Calculate cosine similarities
        inner_products = (query_embedding @ document_embeddings.T).squeeze()

        order = torch.argsort(inner_products, descending=True).cpu().numpy()
        actual = list(np.array(documents)[order])

        expected = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        assert actual == expected, actual

    # python -m unittest semantic_search_models.wrapped_mpnet_test.TestWrappedMpnetModel.test_overfit -v
    def test_overfit(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        model = WrappedMpnetModel(config)

        optimizer = optim.AdamW(model.get_all_trainable_parameters(), lr=1e-5)

        loss_fn = torch.nn.MSELoss()

        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        target_similarities = torch.tensor([1.0, 0.75, 0.5, 0.25], device="cuda:0")

        for epoch in range(20):
            optimizer.zero_grad()

            # Embed the query and the documents
            all_embeddings = model.embed([query, *documents])
            query_embedding = all_embeddings[0]
            document_embeddings = all_embeddings[1:]

            # Calculate cosine similarities
            inner_products = (query_embedding @ document_embeddings.T).squeeze()

            # Calculate loss
            loss = loss_fn(inner_products, target_similarities)

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # python -m unittest semantic_search_models.wrapped_mpnet_test.TestWrappedMpnetModel.test_compute_quality_diversity -v
    def test_compute_quality_diversity(self):
        queries = [
            "What is Alice's favourite fruit?",
            "What is Bob's favourite fruit?",
            "What is Charlie's favourite fruit?",
        ]

        retrieved_documents = [
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Charlie's favourite fruit?",
                ],
            },
            {
                "prompts": [
                    "What is Charlie's favourite fruit?",
                    "What is Alice's favourite fruit?",
                ],
            },
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Alice's favourite fruit?",
                ],
            },
        ]

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        model = WrappedMpnetModel(config)
        question_embeddings = model.embed(queries)
        quality_vector, diversity_matrix = model.compute_quality_diversity(
            question_embeddings, retrieved_documents
        )

        assert quality_vector.shape == torch.Size([3, 2]), quality_vector.shape
        assert diversity_matrix.shape == torch.Size([3, 2, 2]), diversity_matrix.shape


if __name__ == "__main__":
    unittest.main()
