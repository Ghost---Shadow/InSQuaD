import torch


class FlatCutoffStrategy:
    NAME = "flat_cutoff"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.gain_cutoff = (
            self.config.architecture.subset_selection_strategy.gain_cutoff
        )
        assert (
            self.gain_cutoff
        ), f"architecture.subset_selection_strategy.gain_cutoff not initialized {self.gain_cutoff}"

    def subset_select(
        self, query_embedding: torch.Tensor, shortlist_embeddings: torch.Tensor
    ):
        # Assert if query_embedding is 1D and shortlist_embeddings is 2D
        assert (
            query_embedding.ndim == 1
        ), f"query_embedding.ndim {query_embedding.ndim} != 1"
        assert (
            shortlist_embeddings.ndim == 2
        ), f"shortlist_embeddings.ndim {shortlist_embeddings.ndim} != 2"

        query_embedding = query_embedding.unsqueeze(0)

        # Embeddings are already normalized, do matrix multiplication to get cosine similarity
        similarity = torch.matmul(query_embedding, shortlist_embeddings.T).squeeze()

        # Pick indices with similarity > self.gain_cutoff
        picked_indices = torch.nonzero(similarity > self.gain_cutoff).squeeze()

        if picked_indices.ndim < 1:
            picked_indices = picked_indices.unsqueeze(0)

        return picked_indices
