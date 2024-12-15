from subset_selection_strategies.base_strategy import BaseSubsetSelectionStrategy
import torch
from tqdm import tqdm


class GraphCutSubsetStrategy(BaseSubsetSelectionStrategy):
    NAME = "graph_cut"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.k = config.architecture.subset_selection_strategy.k
        self.number_to_select = self.config.offline_validation.annotation_budget

        assert self.k == self.number_to_select, "You need to set k as annotation budget"

    def cosine_similarity_matrix(self, embeddings):
        # Normalize the embeddings to unit length
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        return similarity_matrix

    def graph_cut_score(self, similarity_matrix, selected_indices, candidate_index):
        # Similarity within the selected set
        if selected_indices:
            intra_sim = similarity_matrix[selected_indices][:, selected_indices].sum()
        else:
            intra_sim = 0.0

        # Dissimilarity between selected and candidate
        inter_sim = (
            similarity_matrix[selected_indices, candidate_index].sum()
            if selected_indices
            else 0.0
        )

        # Compute the graph cut score
        return intra_sim - inter_sim

    def subset_select(self, *args):
        if len(args) == 2:
            _, candidate_embeddings = args
        elif len(args) == 1:
            candidate_embeddings = args[0]
        else:
            raise ValueError(f"Expected candidate_embeddings got {args}")

        number_to_select = min(self.number_to_select, len(candidate_embeddings))

        num_docs = candidate_embeddings.shape[0]
        # Compute the cosine similarity matrix
        similarity_matrix = self.cosine_similarity_matrix(candidate_embeddings)

        # Initialize the list of selected indices
        selected_indices = []
        remaining_indices = list(range(num_docs))

        # Initialize scores list for debugging or further analysis
        scores = []

        # Greedy algorithm to select the budget number of facilities
        for _ in tqdm(range(number_to_select), desc="Selecting facilities"):
            best_score = -float("inf")
            best_candidate = None

            for candidate in remaining_indices:
                # Compute the graph cut score for this candidate
                score = self.graph_cut_score(
                    similarity_matrix, selected_indices, candidate
                )

                # Update the best candidate if this one is better
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            # Update the selected and remaining lists
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
            scores.append(best_score)

        selected_indices = torch.tensor(selected_indices)
        scores = torch.tensor(scores)

        return selected_indices, scores
