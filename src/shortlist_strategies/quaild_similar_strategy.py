from collections import Counter
import json
from dataloaders.in_memory import InMemoryDataset
import numpy as np
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm
from config import RootConfig


class QuaildSimilarStrategy(BaseStrategy):
    NAME = "quaild_similar"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, use_cache=True):
        counter = Counter()
        longlist_rows = self.subsample_dataset_for_train()

        cache_name = "long_list.index"
        wrapped_longlist_dataset = InMemoryDataset(self.config, longlist_rows)
        self._populate_and_cache_index(cache_name, use_cache, wrapped_longlist_dataset)

        # Counting votes
        for row in tqdm(longlist_rows, desc="Counting votes"):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            batch = self.pipeline.dense_index.retrieve(prompt_embedding, omit_self=True)
            shortlist_indices = np.array(batch[0]["global_indices"])
            shortlist_prompts = batch[0]["prompts"]

            # TODO: Optimization: Recover embeddings from faiss instead of recomputing
            # but this is better anyways so maybe not?
            shortlist_embeddings = self.pipeline.semantic_search_model.embed(
                shortlist_prompts
            )

            local_shortlist_indices, _ = (
                self.pipeline.subset_selection_strategy.subset_select(
                    prompt_embedding, shortlist_embeddings
                )
            )

            voted_global_indices = shortlist_indices[local_shortlist_indices].tolist()

            if not isinstance(voted_global_indices, list):
                voted_global_indices = [voted_global_indices]

            for idx in list(voted_global_indices):
                counter[idx] += 1

        counter_result = counter.most_common(self.top_n)
        indexes = [item[0] for item in counter_result]
        confidences = [item[1] for item in counter_result]
        confidences = (np.array(confidences) / max(confidences)).tolist()
        return indexes, confidences

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
            candidate_fewshot = self.pipeline.dense_index.retrieve(
                prompt_embedding, omit_self=False
            )
            candidate_fewshot = candidate_fewshot[0]  # batch size 1

            num_shots = self.config.offline_validation.num_shots

            few_shots = {"prompts": [], "labels": []}
            for idx in range(num_shots):
                few_shots["prompts"].append(candidate_fewshot["prompts"][idx])
                few_shots["labels"].append(candidate_fewshot["labels"][idx])

            yield row, few_shots
