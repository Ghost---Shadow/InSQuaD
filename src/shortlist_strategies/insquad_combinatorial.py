import json
import torch
from shortlist_strategies.base import BaseStrategy
from tqdm import tqdm
from config import RootConfig
from dataloaders.in_memory import InMemoryDataset


class InsquadCombinatorialStrategy(BaseStrategy):
    NAME = "insquad_combinatorial"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)
        assert (
            config.architecture.subset_selection_strategy.k
            == config.offline_validation.annotation_budget
        )

    def _cache_similarities(self, use_cache):
        longlist_rows = self.subsample_dataset_for_train()

        if use_cache and self.pipeline.longlist_similarity_tensor_path.exists():
            similarities = torch.load(self.pipeline.longlist_similarity_tensor_path)
        else:
            longlist_embeddings = []
            for row in tqdm(longlist_rows, desc="Generating longlist embeddings"):
                prompt = [row["prompts"]]
                prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
                longlist_embeddings.append(prompt_embedding.squeeze())

            longlist_embeddings = torch.stack(longlist_embeddings)
            similarities = self.pipeline.loss_function.compute_similarity_matrix(
                longlist_embeddings, longlist_embeddings
            )
            torch.save(similarities, self.pipeline.longlist_similarity_tensor_path)

        return similarities

    def shortlist(self, use_cache=True):
        similarities = self._cache_similarities(use_cache)

        # For long list, there is no distinction
        query_query_similarity = similarities
        doc_query_similarity = similarities
        doc_doc_similarity = similarities

        local_shortlist_indices, confidences = (
            self.pipeline.subset_selection_strategy.subset_select_with_similarity(
                query_query_similarity, doc_query_similarity, doc_doc_similarity
            )
        )

        # Already global
        global_indices = local_shortlist_indices.tolist()
        confidences = confidences.tolist()
        return global_indices, confidences

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
