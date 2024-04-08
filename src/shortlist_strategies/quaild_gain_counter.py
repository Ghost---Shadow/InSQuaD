from collections import Counter
from offline_eval_pipeline import OfflineEvaluationPipeline
from tqdm import tqdm
from config import RootConfig


class QuaildGainCounterStrategy:
    def __init__(self, config: RootConfig, pipeline: OfflineEvaluationPipeline):
        self.config = config
        self.pipeline: OfflineEvaluationPipeline = pipeline
        self.counter = Counter()
        self.top_n = self.config.offline_validation.annotation_budget

    def shortlist(self, dataset_name):
        wrapped_dataset = self.pipeline.validation_dataset_lut[dataset_name]
        # Populate dense index
        self.pipeline.dense_index.repopulate_index(
            wrapped_dataset, self.pipeline.semantic_search_model
        )

        # Counting votes
        for row in tqdm(wrapped_dataset, desc="Counting votes"):
            prompt = row["prompt"]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            shortlist_embeddings, shortlist_indices = self.pipeline.dense_index.search(
                prompt_embedding
            )
            local_shortlist_indices = (
                self.pipeline.subset_selection_strategy.subset_select(
                    prompt_embedding, shortlist_embeddings
                )
            )
            voted_global_indices = shortlist_indices[local_shortlist_indices]

            for idx in list(voted_global_indices):
                self.counter[idx].increment()

        return self.counter.top(self.top_n)

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
