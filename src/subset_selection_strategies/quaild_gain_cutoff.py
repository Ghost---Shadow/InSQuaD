import torch


class QuaildGainCutoffStrategy:
    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.gain_cutoff = (
            self.config.architecture.subset_selection_strategy.gain_cutoff
        )
        assert (
            self.gain_cutoff
        ), f"architecture.subset_selection_strategy.gain_cutoff not initialized {self.gain_cutoff}"
        self.lambdA = self.config.validation.q_d_tradeoff_lambda

    def subset_select(
        self, query_embedding: torch.Tensor, shortlist_embeddings: torch.Tensor
    ):
        query_embedding = query_embedding.unsqueeze(0)
        picked_mask = torch.zeros(len(shortlist_embeddings), dtype=torch.bool)

        gain = torch.tensor(float("Inf"))
        num_picked = 0

        while gain > self.gain_cutoff and num_picked < len(picked_mask):
            candidate_indices = torch.nonzero(~picked_mask)
            picked_embeddings = shortlist_embeddings[picked_mask]

            scores = []

            for candidate_index in candidate_indices:
                # TODO: Optimize: Move quality out of while loop
                candidate_embedding = shortlist_embeddings[candidate_index].unsqueeze(0)
                quality = self.pipeline.loss_function(
                    candidate_embedding, query_embedding
                )

                diversity = torch.tensor(1.0, device=quality.device)
                if num_picked > 0:
                    diversity = 1 - self.pipeline.loss_function(
                        picked_embeddings, candidate_embedding
                    )

                score = torch.log(quality) + self.lambdA * torch.log(diversity)
                scores.append(score)

            scores = torch.stack(scores)

            gain, best_candidate = torch.max(scores, dim=0)
            if gain > self.gain_cutoff:
                picked_mask[best_candidate] = True

        return torch.nonzero(picked_mask)
