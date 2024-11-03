import json
import torch
from dataloaders.in_memory import InMemoryDataset
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm
from config import RootConfig


class InsquadCombinatorialStrategy(BaseStrategy):
    NAME = "insquad_combinatorial"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)
        assert (
            config.architecture.subset_selection_strategy.k
            == config.offline_validation.annotation_budget
        )

    def shortlist(self, use_cache=True):
        longlist_rows = self.subsample_dataset_for_train()
        cache_name = "long_list.index"
        wrapped_longlist_dataset = InMemoryDataset(self.config, longlist_rows)
        self._populate_and_cache_index(cache_name, use_cache, wrapped_longlist_dataset)

        query_embeddings = []
        document_embeddings = []
        document_indices = []

        for row in tqdm(longlist_rows, desc="Retrieving similar documents"):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            query_embeddings.append(prompt_embedding)

            batch = self.pipeline.dense_index.retrieve(prompt_embedding, omit_self=True)
            shortlist_indices = torch.tensor(batch[0]["global_indices"])
            shortlist_prompts = batch[0]["prompts"]

            assert max(shortlist_indices) <= len(
                longlist_rows
            ), f"{max(shortlist_indices)} {len(longlist_rows)}"

            shortlist_embeddings = self.pipeline.semantic_search_model.embed(
                shortlist_prompts
            )
            # Should be flatened. Grouping does not matter
            document_embeddings.extend(shortlist_embeddings)
            document_indices.extend(shortlist_indices)

        query_embeddings = torch.stack(query_embeddings)
        document_embeddings = torch.stack(document_embeddings)
        document_indices = torch.stack(document_indices)

        local_shortlist_indices, confidences = (
            self.pipeline.subset_selection_strategy.subset_select(
                query_embeddings, document_embeddings
            )
        )
        global_indices = document_indices[local_shortlist_indices].tolist()
        return global_indices, confidences

    def assemble_few_shot(self, use_cache=True):
        with open(self.pipeline.shortlisted_data_path) as f:
            shortlist = json.load(f)

        eval_list_rows = self.subsample_dataset_for_eval()

        query_embeddings = []
        document_embeddings = []
        document_indices = []

        for row in tqdm(eval_list_rows, desc="Embedding queries"):
            prompt = [row["prompts"]]
            prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
            query_embeddings.append(prompt_embedding)

        for row in tqdm(shortlist, desc="Embedding documents"):
            prompt = [row["prompts"]]
            document_embedding = self.pipeline.semantic_search_model.embed(prompt)
            document_embeddings.append(document_embedding)

        query_embeddings = torch.stack(query_embeddings)
        document_embeddings = torch.stack(document_embeddings)
        document_indices = torch.arange(len(shortlist))

        local_shortlist_indices, _ = (
            self.pipeline.subset_selection_strategy.subset_select(
                query_embeddings, document_embeddings
            )
        )
        global_indices = document_indices[local_shortlist_indices].tolist()

        num_shots = self.config.offline_validation.num_shots
        fewshot_indices = global_indices[:num_shots]

        assert len(fewshot_indices) == num_shots, len(fewshot_indices)

        few_shots = {"prompts": [], "labels": []}
        for idx in fewshot_indices:
            few_shots["prompts"].append(shortlist[idx]["prompts"])
            few_shots["labels"].append(shortlist[idx]["labels"])

        # Same few shot for all
        for row in tqdm(eval_list_rows, desc="Assembling few shot"):
            yield row, few_shots
