from collections import Counter
import json
from dataloaders.in_memory import InMemoryDataset
import numpy as np
from tqdm import tqdm
from config import RootConfig


class QuaildGainCounterStrategy:
    def __init__(self, config: RootConfig, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.counter = Counter()
        self.top_n = self.config.offline_validation.annotation_budget

    def subsample_dataset(self, wrapped_dataset):
        # TODO: Shuffle the subsample
        dataset_length = len(wrapped_dataset.dataset["train"])
        subsample_for_eval_size = self.config.offline_validation.subsample_for_eval_size

        total = min(dataset_length, subsample_for_eval_size)

        def _iterator():
            i = 0
            for row in wrapped_dataset:
                yield row
                i += 1
                if i == total:
                    break

        return total, _iterator()

    def _populate_and_cache_index(self, cache_name, use_cache, wrapped_dataset):
        if use_cache and self.pipeline.dense_index.does_cache_exist(cache_name):
            print("Found dense index cache")
            self.pipeline.dense_index.load_index(wrapped_dataset, cache_name)
        else:
            # Populate dense index
            self.pipeline.dense_index.repopulate_index(
                wrapped_dataset,
                self.pipeline.semantic_search_model,
            )
            self.pipeline.dense_index.save_index(cache_name)

    def shortlist(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]
        cache_name = "long_list.index"
        self._populate_and_cache_index(cache_name, use_cache, wrapped_dataset)

        # Counting votes
        total, subsampled_dataset = self.subsample_dataset(wrapped_dataset)
        for row in tqdm(subsampled_dataset, total=total, desc="Counting votes"):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            batch = self.pipeline.dense_index.retrieve(prompt_embedding)
            shortlist_indices = np.array(batch[0]["global_indices"])
            shortlist_prompts = batch[0]["prompts"]

            # TODO: Optimization: Recover embeddings from faiss instead of recomputing
            # but this is better anyways so maybe not?
            shortlist_embeddings = self.pipeline.semantic_search_model.embed(
                shortlist_prompts
            )

            local_shortlist_indices = (
                self.pipeline.subset_selection_strategy.subset_select(
                    prompt_embedding, shortlist_embeddings
                )
            )

            voted_global_indices = shortlist_indices[local_shortlist_indices].tolist()

            if not isinstance(voted_global_indices, list):
                voted_global_indices = [voted_global_indices]

            for idx in list(voted_global_indices):
                self.counter[idx] += 1

        counter_result = self.counter.most_common(self.top_n)
        indexes = [item[0] for item in counter_result]
        confidences = [item[1] for item in counter_result]
        confidences = (np.array(confidences) / max(confidences)).tolist()
        return indexes, confidences

    def assemble_few_shot(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]

        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        # Populate dense index with shortlist this time
        wrapped_shortlist_dataset = InMemoryDataset(self.config, shortlist)
        cache_name = "short_list.index"
        self._populate_and_cache_index(cache_name, use_cache, wrapped_shortlist_dataset)
        validation_dataset = wrapped_dataset.get_row_iterator("validation")
        for row in tqdm(
            validation_dataset,
            desc="Assembling few shot",
            total=len(wrapped_dataset.dataset["validation"]),
        ):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            candidate_fewshot = self.pipeline.dense_index.retrieve(prompt_embedding)
            candidate_fewshot = candidate_fewshot[0]  # batch size 1
            candidate_fewshot_indices = candidate_fewshot["global_indices"]
            candidate_fewshot_prompts = candidate_fewshot["prompts"]

            # TODO: Optimization: Recover embeddings from faiss instead of recomputing
            # but this is better anyways so maybe not?
            candidate_fewshot_embeddings = self.pipeline.semantic_search_model.embed(
                candidate_fewshot_prompts
            )

            local_fewshot_indices = (
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
