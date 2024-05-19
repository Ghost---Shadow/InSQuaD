import torch
import torch.nn.functional as F


def bind_pick_most_diverse(self):
    device = self.loss_fn.config.architecture.semantic_search_model.device

    def _pick_most_diverse(picked_set, candidate_set):
        picked_set = torch.tensor([picked_set], device=device)
        best_diversity = -1
        best_candidate = None
        for candidate in candidate_set:
            candidate_set = torch.tensor([[candidate]], device=device)
            picked_set = F.normalize(picked_set, dim=-1)
            candidate_set = F.normalize(candidate_set, dim=-1)
            diversity = 1 - self.loss_fn.similarity(picked_set, candidate_set)
            # print(candidate, diversity.item())
            if diversity > best_diversity:
                best_diversity = diversity.item()
                best_candidate = candidate
        return best_diversity, best_candidate

    return _pick_most_diverse
