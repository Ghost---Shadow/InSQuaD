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
            self.gain_cutoff is not None
        ), f"architecture.subset_selection_strategy.gain_cutoff not initialized {self.gain_cutoff}"
        self.lambdA = self.config.offline_validation.q_d_tradeoff_lambda
        self.epsilon = 1e-7

    def subset_select(
        self, query_embedding: torch.Tensor, shortlist_embeddings: torch.Tensor
    ):
        query_embedding = query_embedding.unsqueeze(0)
        picked_mask = torch.zeros(len(shortlist_embeddings), dtype=torch.bool)
        gains = []

        for _ in range(len(shortlist_embeddings)):
            candidate_indices = torch.nonzero(~picked_mask).squeeze(dim=-1)
            picked_embeddings = shortlist_embeddings[picked_mask]
            scores = []

            for candidate_index in candidate_indices:
                candidate_embedding = shortlist_embeddings[candidate_index].unsqueeze(0)
                quality = self.pipeline.loss_function.similarity(
                    candidate_embedding, query_embedding
                )

                diversity = torch.tensor(1.0, device=quality.device)
                if picked_mask.sum() > 0:
                    diversity = 1 - self.pipeline.loss_function.similarity(
                        picked_embeddings, candidate_embedding
                    )
                log_quality = torch.log(quality + self.epsilon)
                log_diversity = torch.log(diversity + self.epsilon)
                score = (1 - self.lambdA) * log_quality + self.lambdA * log_diversity
                score = torch.exp(score).item()
                scores.append((candidate_index, score))
            #     s = f"{candidate_index.item()}\t{score:.4f}\t{quality.item():.4f}\t{diversity.item():.4f}"
            #     print(s)

            # print("-" * 80)

            best_candidate, gain = max(scores, key=lambda x: x[1])
            if gain > self.gain_cutoff:
                picked_mask[best_candidate] = True
                gains.append((best_candidate.item(), gain))
            else:
                break

        # Sort indices by gain, in descending order
        gains.sort(key=lambda x: x[1], reverse=True)
        # print(gains)
        picked_indices = torch.tensor([x[0] for x in gains])
        gains = torch.tensor([x[1] for x in gains])

        return picked_indices, gains
