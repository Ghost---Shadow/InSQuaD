from collections import Counter
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

    def shortlist(self, dataset_name, use_cache=True):
        wrapped_dataset = self.pipeline.offline_dataset_lut[dataset_name]
        if use_cache and self.pipeline.dense_index.does_cache_exist:
            print("Found dense index cache")
            self.pipeline.dense_index.load_index(wrapped_dataset)
        else:
            # Populate dense index
            self.pipeline.dense_index.repopulate_index(
                wrapped_dataset,
                self.pipeline.semantic_search_model,
            )
            self.pipeline.dense_index.save_index()

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

    def assemble_few_shot(self, wrapped_dataset, shortlist):
        # Populate dense index with shortlist this time
        self.pipeline.dense_index.repopulate_index(
            shortlist, self.pipeline.semantic_search_model
        )

        for row in tqdm(wrapped_dataset, desc="Assembling few shot"):
            prompt = row["prompt"]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            few_shot_embeddings, few_shot_indices = self.pipeline.dense_index.search(
                prompt_embedding
            )

            local_fewer_shot_indices = (
                self.pipeline.subset_selection_strategy.subset_select(
                    prompt_embedding, few_shot_embeddings
                )
            )

            voted_few_shot_indices = few_shot_indices[local_fewer_shot_indices]
            voted_few_shot_indices = voted_few_shot_indices[: self.pipeline.num_shots]

            few_shots = []
            for idx in voted_few_shot_indices:
                few_shots.append(shortlist[idx])

            yield row, few_shots
