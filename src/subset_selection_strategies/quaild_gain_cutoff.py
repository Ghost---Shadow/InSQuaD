import torch


class QuaildGainCutoffStrategy:
    NAME = "gain_cutoff"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.gain_cutoff = (
            self.config.architecture.subset_selection_strategy.gain_cutoff
        )
        assert (
            self.gain_cutoff
        ), f"architecture.subset_selection_strategy.gain_cutoff not initialized {self.gain_cutoff}"
        self.lambdA = self.config.offline_validation.q_d_tradeoff_lambda

    def subset_select(
        self, query_embedding: torch.Tensor, shortlist_embeddings: torch.Tensor
    ):
        query_embedding = query_embedding.unsqueeze(0)
        picked_mask = torch.zeros(len(shortlist_embeddings), dtype=torch.bool)
        gain = torch.tensor(float("Inf"))

        while gain > self.gain_cutoff and picked_mask.sum() < len(picked_mask):
            candidate_indices = torch.nonzero(~picked_mask).squeeze(dim=-1)
            picked_embeddings = shortlist_embeddings[picked_mask]
            scores = []

            for candidate_index in candidate_indices:
                # TODO: Optimize: Move quality out of while loop
                candidate_embedding = shortlist_embeddings[candidate_index].unsqueeze(0)
                quality = self.pipeline.loss_function.similarity(
                    candidate_embedding, query_embedding
                )

                diversity = torch.tensor(1.0, device=quality.device)
                if picked_mask.sum() > 0:
                    diversity = 1 - self.pipeline.loss_function.similarity(
                        picked_embeddings, candidate_embedding
                    )
                log_quality = torch.log(quality)
                log_diversity = torch.log(diversity)
                score = 2 * (
                    (1 - self.lambdA) * log_quality + self.lambdA * log_diversity
                )
                score = torch.exp(score).item()
                scores.append((candidate_index, score))

            best_candidate, gain = max(scores, key=lambda x: x[1])
            if gain > self.gain_cutoff:
                picked_mask[best_candidate] = True

        picked_indices = torch.nonzero(picked_mask)
        if len(picked_indices.shape) == 2:
            picked_indices = picked_indices.squeeze(dim=-1)

        assert len(picked_indices.shape) == 1, picked_indices

        return picked_indices
