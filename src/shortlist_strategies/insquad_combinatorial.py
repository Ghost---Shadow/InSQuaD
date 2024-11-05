import json
import torch
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

        eval_list_rows = self.subsample_dataset_for_eval()

        if (
            use_cache
            and self.pipeline.shortlist_qq_similarity_tensor_path.exists()
            and self.pipeline.shortlist_dq_similarity_tensor_path.exists()
            and self.pipeline.shortlist_dd_similarity_tensor_path.exists()
        ):
            query_query_similarity = torch.load(
                self.pipeline.shortlist_qq_similarity_tensor_path
            )
            doc_query_similarity = torch.load(
                self.pipeline.shortlist_dq_similarity_tensor_path
            )
            doc_doc_similarity = torch.load(
                self.pipeline.shortlist_dd_similarity_tensor_path
            )
        else:
            query_embeddings = []
            document_embeddings = []

            for row in tqdm(eval_list_rows, desc="Embedding queries"):
                prompt = [row["prompts"]]
                prompt_embedding = self.pipeline.semantic_search_model.embed(prompt)
                query_embeddings.append(prompt_embedding)

            for row in tqdm(shortlist, desc="Embedding documents"):
                prompt = [row["prompts"]]
                document_embedding = self.pipeline.semantic_search_model.embed(prompt)
                document_embeddings.append(document_embedding)

            query_embeddings = torch.stack(query_embeddings, dim=1)
            document_embeddings = torch.stack(document_embeddings, dim=1)

            query_query_similarity = (
                self.pipeline.loss_function.compute_similarity_matrix(
                    query_embeddings, query_embeddings
                )
            )

            doc_query_similarity = (
                self.pipeline.loss_function.compute_similarity_matrix(
                    document_embeddings, query_embeddings
                )
            )

            doc_doc_similarity = self.pipeline.loss_function.compute_similarity_matrix(
                document_embeddings, document_embeddings
            )

            torch.save(
                query_query_similarity,
                self.pipeline.shortlist_qq_similarity_tensor_path,
            )
            torch.save(
                doc_query_similarity,
                self.pipeline.shortlist_dq_similarity_tensor_path,
            )
            torch.save(
                doc_doc_similarity,
                self.pipeline.shortlist_dd_similarity_tensor_path,
            )

        local_shortlist_indices, _ = (
            self.pipeline.subset_selection_strategy.subset_select_with_similarity(
                query_query_similarity, doc_query_similarity, doc_doc_similarity
            )
        )

        # Already global
        global_indices = local_shortlist_indices.tolist()
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
