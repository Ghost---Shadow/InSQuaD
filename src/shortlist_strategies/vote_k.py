import json
from dataloaders.in_memory import InMemoryDataset
from eval_utils import flatten_batch_of_batches
import numpy as np
from shortlist_strategies.base import BaseStrategy
import torch
from tqdm import tqdm


class VoteKStrategy(BaseStrategy):
    NAME = "vote_k"

    def __init__(self, config, pipeline):
        super().__init__(config, pipeline)
        # TODO: unhardcode
        # https://github.com/xlang-ai/icl-selective-annotation/blob/e114472cc620c022e1981e1b85101ae492a0c39a/main.py#L37
        self.vote_k_batch_size = 10

    def generate_embedding_matrix(self, longlist_rows):
        embedding_matrix = []
        for row in tqdm(longlist_rows, desc="Generating embeddings"):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            embedding_matrix.append(prompt_embedding.squeeze().detach().cpu())

        embedding_matrix = torch.stack(embedding_matrix)

        return embedding_matrix

    @staticmethod
    def drop_exact_duplicates(batch_results, batch_query_prompts):
        # This set will contain the prompts that are considered duplicates
        duplicates = set(batch_query_prompts)

        # Collect indices of non-duplicate prompts
        non_duplicate_indices = [
            i
            for i, prompt in enumerate(batch_results["prompts"])
            if prompt not in duplicates
        ]

        # Update batch_results by filtering out duplicates across all keys
        for key in batch_results:
            batch_results[key] = [batch_results[key][i] for i in non_duplicate_indices]

        return batch_results

    def shortlist(self, use_cache=True):
        longlist_rows = self.subsample_dataset_for_train()

        cache_name = "long_list.index"
        wrapped_longlist_dataset = InMemoryDataset(self.config, longlist_rows)
        self._populate_and_cache_index(cache_name, use_cache, wrapped_longlist_dataset)

        embedding_matrix = self.generate_embedding_matrix(longlist_rows)

        # Shortlist with fast vote k
        query_indexes, _ = self.pipeline.subset_selection_strategy.subset_select(
            embedding_matrix,
            number_to_select=self.vote_k_batch_size,
        )

        batch_query_prompts = []
        for idx, row in enumerate(longlist_rows):
            if idx in query_indexes:
                batch_query_prompts.append(row["prompts"])

        # batch_prompt_embeddings = train_embs from original implementation
        # self.pipeline.dense_index is preloaded with "test_embs" (actually train) from original implementation
        # https://github.com/xlang-ai/icl-selective-annotation/blob/e114472cc620c022e1981e1b85101ae492a0c39a/two_steps.py#L156
        batch_prompt_embeddings = self.pipeline.semantic_search_model.embed(
            batch_query_prompts
        )
        batched_similar_result = self.pipeline.dense_index.retrieve(
            batch_prompt_embeddings
        )
        batch_results = flatten_batch_of_batches(batched_similar_result)
        batch_results = self.drop_exact_duplicates(batch_results, batch_query_prompts)

        # "Take as much as it fits in context" is handled downstream
        indexes = np.array(batch_results["global_indices"])
        distances = np.array(batch_results["distances"])

        # Deduplicate
        indexes, unique_index_indices = np.unique(indexes, return_index=True)
        distances = distances[unique_index_indices]

        # Distance is bad, so sort ascending
        order = np.argsort(distances)
        indexes = indexes[order]
        distances = distances[order]

        budget = self.config.offline_validation.annotation_budget
        indexes = indexes[:budget]
        distances = distances[:budget]
        scores = 1 - distances

        indexes = indexes.tolist()
        scores = scores.tolist()

        return indexes, scores

    def assemble_few_shot(self, use_cache=True):
        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        # Populate dense index with shortlist
        wrapped_shortlist_dataset = InMemoryDataset(self.config, shortlist)
        cache_name = "short_list.index"
        self._populate_and_cache_index(cache_name, use_cache, wrapped_shortlist_dataset)

        eval_list_rows = self.subsample_dataset_for_eval()

        for row in tqdm(eval_list_rows, desc="Assembling few shot"):
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
