from subset_selection_strategies.base_strategy import BaseSubsetSelectionStrategy
import torch
from tqdm import tqdm
import random


class DiversitySubsetSelectionStrategy(BaseSubsetSelectionStrategy):
    NAME = "diversity"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.k = config.architecture.subset_selection_strategy.k
        self.number_to_select = self.config.offline_validation.annotation_budget

        assert self.k == self.number_to_select, "You need to set k as annotation budget"

    def subset_select(self, *args):
        if len(args) == 2:
            _, candidate_embeddings = args
        elif len(args) == 1:
            candidate_embeddings = args[0]
        else:
            raise ValueError(f"Expected candidate_embeddings got {args}")

        number_to_select = min(self.number_to_select, len(candidate_embeddings))

        if number_to_select == 0:
            indices = torch.tensor([], dtype=torch.long)
            scores = torch.tensor([], dtype=torch.float32)
            return indices, scores

        # Ensure all tensors are on the same device as candidate_embeddings
        device = candidate_embeddings.device
        selected_indices = []
        all_scores = []

        # Randomly select the first index
        first_id = random.choice(range(len(candidate_embeddings)))
        selected_indices.append(first_id)

        # Initialize selected representations on the same device
        selected_representations = (
            candidate_embeddings[first_id].unsqueeze(0).to(device)
        )

        for _ in tqdm(range(number_to_select - 1)):
            scores = torch.zeros(len(candidate_embeddings), device=device)

            # Calculate cosine similarity for each selected representation
            for selected in selected_representations:
                scores += torch.nn.functional.cosine_similarity(
                    candidate_embeddings, selected.unsqueeze(0), dim=1, eps=1e-8
                )

            # Set already selected indices' scores to a large number to exclude them
            for i in selected_indices:
                scores[i] = float("inf")

            # Find the index with the minimum score (highest diversity)
            min_idx = torch.argmin(scores).item()
            selected_representations = torch.cat(
                (
                    selected_representations,
                    candidate_embeddings[min_idx].unsqueeze(0).to(device),
                ),
                0,
            )
            selected_indices.append(min_idx)
            all_scores.append(scores[min_idx].item())

        # Finalize the scores list to only include scores of selected indices
        final_scores = [all_scores[i] for i in range(len(all_scores))]
        final_scores = [0.0, *final_scores]  # Diversity for 1st element

        selected_indices = torch.tensor(selected_indices)
        final_scores = torch.tensor(final_scores)

        return selected_indices, final_scores
