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
            self.gain_cutoff is not None
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

        # Expand query_embedding to 2D for matrix multiplication
        query_embedding = query_embedding.unsqueeze(0)

        # Calculate cosine similarity
        similarity = torch.matmul(query_embedding, shortlist_embeddings.T).squeeze()

        # Filter indices based on gain_cutoff
        picked_indices = (similarity > self.gain_cutoff).nonzero(as_tuple=True)[0]

        # Sort picked_indices by similarity, in descending order
        if picked_indices.numel() > 0:
            picked_similarities = similarity[picked_indices]
            sorted_indices = picked_similarities.argsort(descending=True)
            picked_indices = picked_indices[sorted_indices]

        return picked_indices
