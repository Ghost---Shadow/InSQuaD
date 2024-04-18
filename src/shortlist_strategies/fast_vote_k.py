import json
from config import RootConfig
from dataloaders.in_memory import InMemoryDataset
from eval_utils import evaluate_with_options_if_possible, get_options_if_possible
from shortlist_strategies.base import BaseStrategy
import torch
from tqdm import tqdm
from collections import defaultdict
from tqdm import tqdm


class FastVoteKStrategy(BaseStrategy):
    NAME = "fast_vote_k"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)
        # TODO: Unhardcode (https://github.com/xlang-ai/icl-selective-annotation/blob/e114472cc620c022e1981e1b85101ae492a0c39a/two_steps.py#L333)
        self.k = 150

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

    def shortlist(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]
        total, subsampled_train_iterator = self.subsample_dataset(
            wrapped_dataset, "train"
        )

        embedding_matrix = []
        for row in tqdm(
            subsampled_train_iterator, total=total, desc="Generating embeddings"
        ):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            embedding_matrix.append(prompt_embedding.squeeze())

        embedding_matrix = torch.stack(embedding_matrix)
        number_to_select = self.config.offline_validation.annotation_budget
        top_k = self.k

        indexes, confidences = self.fast_vote_k(
            embedding_matrix, number_to_select, top_k
        )

        return indexes, confidences

    def assemble_few_shot(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]

        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        # Populate dense index with shortlist
        wrapped_shortlist_dataset = InMemoryDataset(self.config, shortlist)
        cache_name = "short_list.index"
        self._populate_and_cache_index(cache_name, use_cache, wrapped_shortlist_dataset)

        total, subsampled_validation_iterator = self.subsample_dataset(
            wrapped_dataset, "validation"
        )

        for row in tqdm(
            subsampled_validation_iterator,
            desc="Assembling few shot",
            total=total,
        ):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            candidate_fewshot = self.pipeline.dense_index.retrieve(prompt_embedding)
            candidate_fewshot = candidate_fewshot[0]  # batch size 1

            num_shots = self.config.offline_validation.num_shots

            few_shots = {"prompts": [], "labels": []}
            for idx in range(num_shots):
                few_shots["prompts"].append(candidate_fewshot["prompts"][idx])
                few_shots["labels"].append(candidate_fewshot["labels"][idx])

            yield row, few_shots
