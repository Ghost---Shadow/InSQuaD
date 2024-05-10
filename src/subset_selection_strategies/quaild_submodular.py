import torch


class QuaildSubmodularStrategy:
    NAME = "quaild_submodular"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.epsilon = 1e-7
        self.gain_cutoff = config.architecture.subset_selection_strategy.gain_cutoff
        self.top_k = config.architecture.subset_selection_strategy.k

    def subset_select(
        self, query_embedding: torch.Tensor, shortlist_embeddings: torch.Tensor
    ):
        """
        query_embedding.shape = [embedding_dim]
        shortlist_embeddings.shape = [num_docs, embedding_dim]
        """
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
