import json
import os
from pathlib import Path
from dataloaders.in_memory import InMemoryDataset
from shortlist_strategies.base import BaseStrategy
import torch
from tqdm import tqdm


class ShortlistThenTopK(BaseStrategy):
    NAME = "shortlist_then_top_k"

    def __init__(self, config, pipeline):
        super().__init__(config, pipeline)

    def cached_embedding_matrix(self, longlist_rows, use_cache=True):
        cache_file_name = Path(self.pipeline.artifacts_dir) / "longlist_embeddings.pt"
        if os.path.exists(cache_file_name) and use_cache:
            with open(cache_file_name) as f:
                return torch.load(cache_file_name)

        embedding_matrix = []
        for row in tqdm(longlist_rows, desc="Generating embeddings"):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            embedding_matrix.append(prompt_embedding.squeeze().detach().cpu())

        embedding_matrix = torch.stack(embedding_matrix)

        with open(cache_file_name, "wb") as f:
            torch.save(embedding_matrix, f)

        return embedding_matrix

    def shortlist(self, use_cache=True):
        longlist_rows = self.subsample_dataset_for_train()

        embedding_matrix = self.cached_embedding_matrix(longlist_rows, use_cache)

        # Diversity only strategy
        indexes, confidences = self.pipeline.subset_selection_strategy.subset_select(
            embedding_matrix
        )

        if type(indexes) == torch.Tensor:
            indexes = indexes.tolist()
        if type(confidences) == torch.Tensor:
            confidences = confidences.tolist()

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
