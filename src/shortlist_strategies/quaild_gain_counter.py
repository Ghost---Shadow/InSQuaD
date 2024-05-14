from collections import Counter
import json
from dataloaders.in_memory import InMemoryDataset
import numpy as np
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm
from config import RootConfig


class QuaildGainCounterStrategy(BaseStrategy):
    NAME = "quaild_gain_counter"

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

            assert max(shortlist_indices) <= len(
                longlist_rows
            ), f"{max(shortlist_indices)} {len(longlist_rows)}"

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

            assert max(voted_global_indices) <= len(
                longlist_rows
            ), f"{max(voted_global_indices)} {len(longlist_rows)}"

            for idx in list(voted_global_indices):
                counter[idx] += 1

        counter_result = counter.most_common(self.top_n)
        indexes = [item[0] for item in counter_result]
        confidences = [item[1] for item in counter_result]
        confidences = (np.array(confidences) / max(confidences)).tolist()
        assert max(indexes) <= len(
            longlist_rows
        ), f"{max(indexes)} {len(longlist_rows)}"
        return indexes, confidences

    def assemble_few_shot(self, use_cache=True):
        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        # Populate dense index with shortlist this time
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
            candidate_fewshot_indices = candidate_fewshot["global_indices"]
            candidate_fewshot_prompts = candidate_fewshot["prompts"]

            # TODO: Optimization: Recover embeddings from faiss instead of recomputing
            # but this is better anyways so maybe not?
            candidate_fewshot_embeddings = self.pipeline.semantic_search_model.embed(
                candidate_fewshot_prompts
            )

            local_fewshot_indices, _ = (
                self.pipeline.subset_selection_strategy.subset_select(
                    prompt_embedding, candidate_fewshot_embeddings
                )
            )

            num_shots = self.config.offline_validation.num_shots
            fewshot_indices = [
                candidate_fewshot_indices[i] for i in local_fewshot_indices
            ]
            fewshot_indices = fewshot_indices[:num_shots]

            few_shots = {"prompts": [], "labels": []}
            for idx in fewshot_indices:
                few_shots["prompts"].append(shortlist[idx]["prompts"])
                few_shots["labels"].append(shortlist[idx]["labels"])

            # Add some from the candidate fewshot to get to the n-shot
            # TODO: More principled way
            if len(few_shots["prompts"]) < num_shots:
                # Calculate how many more entries are needed.
                more_required = num_shots - len(few_shots["prompts"])

                # Find indices of prompts not yet in few_shots.
                remaining_indices = [
                    i
                    for i, prompt in enumerate(candidate_fewshot["prompts"])
                    if prompt not in set(few_shots["prompts"])
                ]

                # Ensure only as many elements as needed are added.
                selected_indices = remaining_indices[:more_required]

                # Append missing prompts and corresponding labels to few_shots.
                few_shots["prompts"].extend(
                    candidate_fewshot["prompts"][i] for i in selected_indices
                )
                few_shots["labels"].extend(
                    candidate_fewshot["labels"][i] for i in selected_indices
                )

            yield row, few_shots
