import torch
from tqdm import tqdm


class QuaildSubmodularStrategy:
    NAME = "quaild_submodular"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.epsilon = 1e-7
        self.gain_cutoff = config.architecture.subset_selection_strategy.gain_cutoff
        self.top_k = config.architecture.subset_selection_strategy.k

    def subset_select_with_similarity(
        self,
        query_query_similarity: torch.Tensor,
        doc_query_similarity: torch.Tensor,
        doc_doc_similarity: torch.Tensor,
    ):
        """
        query_query_similarity.shape = [batch_size, num_queries, num_queries]
        doc_query_similarity.shape = [batch_size, num_docs, num_queries]
        doc_doc_similarity.shape = [batch_size, num_docs, num_docs]
        """

        batch_size, num_docs, num_queries = doc_query_similarity.shape

        assert query_query_similarity.shape[1] == num_queries, (
            query_query_similarity.shape[1],
            num_queries,
        )
        assert query_query_similarity.shape[2] == num_queries, (
            query_query_similarity.shape[2],
            num_queries,
        )
        assert doc_doc_similarity.shape[1] == num_docs, (
            doc_doc_similarity.shape[1],
            num_docs,
        )
        assert doc_doc_similarity.shape[2] == num_docs, (
            doc_doc_similarity.shape[2],
            num_docs,
        )

        assert batch_size == 1, batch_size  # TODO

        picked_mask = torch.zeros(num_docs, dtype=torch.bool)
        gains = []

        for _ in tqdm(
            range(min(self.top_k, num_docs)), desc="Picking documents", leave=False
        ):
            picked_indices = torch.nonzero(picked_mask).squeeze(dim=-1)
            candidate_indices = torch.nonzero(~picked_mask).squeeze(dim=-1)

            if picked_mask.sum() == 0:
                current_information = torch.tensor(0.0)
            else:
                doc_query_similarity_picked = doc_query_similarity[:, picked_indices, :]
                doc_doc_similarity_picked = doc_doc_similarity[:, picked_indices, :][
                    :, :, picked_indices
                ]
                current_information = (
                    self.pipeline.loss_function.similarity_matrix_to_information(
                        query_query_similarity,
                        doc_query_similarity_picked,
                        doc_doc_similarity_picked,
                    )
                )

            candidate_gains = []
            for candidate_index in candidate_indices:
                next_indices = torch.cat([picked_indices, candidate_index.unsqueeze(0)])
                doc_query_similarity_next = doc_query_similarity[:, next_indices, :]
                doc_doc_similarity_next = doc_doc_similarity[:, next_indices, :][
                    :, :, next_indices
                ]
                new_information = (
                    self.pipeline.loss_function.similarity_matrix_to_information(
                        query_query_similarity,
                        doc_query_similarity_next,
                        doc_doc_similarity_next,
                    )
                )
                information_gain = new_information - current_information
                candidate_gains.append((candidate_index, information_gain.item()))

            best_candidate, best_gain = max(candidate_gains, key=lambda x: x[1])
            picked_mask[best_candidate] = True
            gains.append((best_candidate.item(), best_gain))

        if len(gains) == 0:
            picked_indices = torch.tensor([], dtype=torch.long)
            scores = torch.tensor([], dtype=torch.float32)
            return picked_indices, scores

        picked_indices, scores = zip(*gains)
        picked_indices = torch.tensor(picked_indices)
        scores = torch.tensor(scores)

        picked_indices = picked_indices[: self.top_k]
        scores = scores[: self.top_k]

        return picked_indices, scores

    def subset_select(
        self, query_embedding: torch.Tensor, shortlist_embeddings: torch.Tensor
    ):
        """
        query_embedding.shape = [embedding_dim] or [num_queries, embedding_dim]
        shortlist_embeddings.shape = [num_docs, embedding_dim]
        """
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        picked_mask = torch.zeros(len(shortlist_embeddings), dtype=torch.bool)
        gains = []

        for _ in range(len(shortlist_embeddings)):
            candidate_indices = torch.nonzero(~picked_mask).squeeze(dim=-1)
            picked_embeddings = shortlist_embeddings[picked_mask]
            if len(picked_embeddings) == 0:
                current_information = torch.tensor(0.0)
            else:
                current_information = self.pipeline.loss_function.similarity(
                    picked_embeddings, query_embedding
                )

            candidate_gains = []

            for candidate_index in candidate_indices:
                candidate_embedding = shortlist_embeddings[candidate_index].unsqueeze(0)
                candidate_and_picked_embeddings = torch.cat(
                    (candidate_embedding, picked_embeddings), dim=0
                )
                new_information = self.pipeline.loss_function.similarity(
                    candidate_and_picked_embeddings, query_embedding
                )
                information_gain = new_information - current_information
                candidate_gains.append((candidate_index, information_gain.item()))

            best_candidate, best_gain = max(candidate_gains, key=lambda x: x[1])
            picked_mask[best_candidate] = True
            gains.append((best_candidate.item(), best_gain))

        if len(gains) == 0:
            picked_indices = torch.tensor([], dtype=torch.long)
            scores = torch.tensor([], dtype=torch.float32)
            return picked_indices, scores

        picked_indices, scores = zip(*gains)
        picked_indices = torch.tensor(picked_indices)
        scores = torch.tensor(scores)

        if self.gain_cutoff is not None:
            mask = scores > self.gain_cutoff
            picked_indices = picked_indices[mask]
            scores = scores[mask]

        if self.top_k is not None:
            picked_indices = picked_indices[: self.top_k]
            scores = scores[: self.top_k]

        return picked_indices, scores
