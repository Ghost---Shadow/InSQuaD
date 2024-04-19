from collections import defaultdict
from subset_selection_strategies.base_strategy import BaseSubsetSelectionStrategy
import torch
from tqdm import tqdm


class FastVoteKSubsetStrategy(BaseSubsetSelectionStrategy):
    NAME = "fast_vote_k"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.k = config.architecture.subset_selection_strategy.k
        self.number_to_select = self.config.offline_validation.annotation_budget

    def subset_select(self, embedding_matrix, number_to_select=None):
        if number_to_select is None:
            number_to_select = self.number_to_select

        return FastVoteKSubsetStrategy.fast_vote_k(
            embedding_matrix, number_to_select, self.k
        )

    @staticmethod
    def fast_vote_k(embedding_matrix, number_to_select, top_k):
        """
        Borrowed from
        https://github.com/xlang-ai/icl-selective-annotation/blob/e114472cc620c022e1981e1b85101ae492a0c39a/two_steps.py#L99
        then asked ChatGPT to clean it up

        Selects a subset of embeddings based on a voting mechanism and returns their scores.

        Args:
        embedding_matrix (list): A list of embeddings (numpy arrays).
        number_to_select (int): The number of embeddings to select based on votes.
        top_k (int): The number of similar embeddings to consider for each voting.
        votes_file_path (str, optional): Path to a file to save/load voting results.

        Returns:
        tuple: A tuple containing two lists:
            - Indices of the selected embeddings.
            - Scores of the selected embeddings.
        """

        num_embeddings = len(embedding_matrix)

        voting_stats = defaultdict(list)

        # Compute cosine similarity and gather votes
        for i in tqdm(range(num_embeddings), desc="Voting process part 1"):
            # Reshape to (1, -1)
            current_embedding = embedding_matrix[i].unsqueeze(0)
            similarity_scores = torch.nn.functional.cosine_similarity(
                embedding_matrix, current_embedding, dim=1
            )
            # Exclude self index
            top_similar_indices = torch.topk(
                similarity_scores, top_k + 1, largest=True, sorted=True
            ).indices[:-1]

            for idx in top_similar_indices:
                if idx != i:
                    voting_stats[idx.item()].append(i)

        selected_indices = []
        selected_scores = []
        selection_count = defaultdict(int)

        pbar = tqdm(range(number_to_select), desc="Voting process part 2")

        # Select indices based on voting until the required number is selected
        while len(selected_indices) < number_to_select:
            candidate_scores = defaultdict(int)

            for idx, supporters in voting_stats.items():
                if idx in selected_indices:
                    continue  # Skip already selected indices
                for supporter in supporters:
                    if supporter not in selected_indices:
                        # Dynamic score adjustment
                        candidate_scores[idx] += 10 ** (-selection_count[supporter])

            # Get the index and score of the candidate with the highest score
            next_selected_index, next_score = max(
                candidate_scores.items(), key=lambda item: item[1]
            )
            selected_indices.append(next_selected_index)
            selected_scores.append(next_score)
            pbar.update(1)

            # Update the selection count for supporters
            for supporter in voting_stats[next_selected_index]:
                selection_count[supporter] += 1

        return selected_indices, selected_scores
