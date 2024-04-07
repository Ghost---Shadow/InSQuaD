from collections import Counter
from tqdm import tqdm
from config import Config
from training_pipeline import TrainingPipeline


class QuaildGainCounterStrategy:
    def __init__(self, config: Config, pipeline: TrainingPipeline):
        self.config = config
        self.pipeline = pipeline
        self.counter = Counter()
        self.pipeline.dense_index.reset()
        self.top_n = self.config.validation.annotation_budget

    def shortlist(self):
        long_list_loader = self.pipeline.long_list_loader
        # Populate dense index
        # TODO: Cache
        for row in tqdm(long_list_loader, desc="Populating dense index"):
            prompt = row["prompt"]
            prompt_embedding = self.pipeline.embedding_model.embed(prompt)
            self.pipeline.dense_index.add(prompt_embedding)

        # Counting votes
        for row in tqdm(long_list_loader, desc="Counting votes"):
            prompt = row["prompt"]
            prompt_embedding = self.pipeline.embedding_model.embed(prompt)
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
