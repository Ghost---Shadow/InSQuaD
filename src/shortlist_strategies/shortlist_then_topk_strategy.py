import json
from dataloaders.in_memory import InMemoryDataset
from shortlist_strategies.base import BaseStrategy
import torch
from tqdm import tqdm


class ShortlistThenTopK(BaseStrategy):
    NAME = "shortlist_then_top_k"

    def __init__(self, config, pipeline):
        super().__init__(config, pipeline)

    def shortlist(self, use_cache=True):
        longlist_rows = self.subsample_dataset_for_train()

        embedding_matrix = []
        for row in tqdm(longlist_rows, desc="Generating embeddings"):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            embedding_matrix.append(prompt_embedding.squeeze().detach().cpu())

        embedding_matrix = torch.stack(embedding_matrix)

        # Diversity only strategy
        indexes, confidences = self.pipeline.subset_selection_strategy.subset_select(
            embedding_matrix
        )

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
            candidate_fewshot = self.pipeline.dense_index.retrieve(prompt_embedding)
            candidate_fewshot = candidate_fewshot[0]  # batch size 1

            num_shots = self.config.offline_validation.num_shots

            few_shots = {"prompts": [], "labels": []}
            for idx in range(num_shots):
                few_shots["prompts"].append(candidate_fewshot["prompts"][idx])
                few_shots["labels"].append(candidate_fewshot["labels"][idx])

            yield row, few_shots
