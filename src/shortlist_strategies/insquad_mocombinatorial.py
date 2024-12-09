import random
from shortlist_strategies.insquad_combinatorial import InsquadCombinatorialStrategy
import torch
from tqdm import tqdm
from config import RootConfig


class InsquadMemoryOptimizedCombinatorialStrategy(InsquadCombinatorialStrategy):
    NAME = "insquad_mocombinatorial"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

        self.window_size = (
            config.offline_validation.insquad_mocombinatorial_config.window_size
        )
        self.annotation_budget = config.offline_validation.annotation_budget
        self.max_chunks_to_process = (
            config.offline_validation.insquad_mocombinatorial_config.max_chunks_to_process
        )

        if (
            config.offline_validation.subsample_for_train_size
            % config.offline_validation.insquad_mocombinatorial_config.window_size
            != 0
        ):
            raise ValueError(
                "offline_validation.insquad_mocombinatorial_config.window_size is not divisible by offline_validation.subsample_for_train_size"
            )

    def shortlist(self, use_cache=True):
        similarities = self._cache_similarities(use_cache)  # [1xnxn square matrix]

        n = similarities.shape[1]
        chunk_size = self.window_size
        num_chunks = n // chunk_size

        # To keep track of top entries
        top_indices = []
        top_confidences = []

        # If not exactly divisible
        if n % self.window_size != 0:
            n = n - n % self.window_size

        flat_list = [
            (row_start, col_start)
            for row_start in range(0, n, chunk_size)
            for col_start in range(0, n, chunk_size)
        ]

        if self.max_chunks_to_process is not None:
            flat_list = random.sample(
                flat_list, min(len(flat_list), self.max_chunks_to_process)
            )

        # Processing each grid block
        for row_start, col_start in tqdm(
            flat_list, leave=False, desc="Sweeping chunks"
        ):
            # Defining the boundaries of the sub-matrix
            row_end = row_start + chunk_size
            col_end = col_start + chunk_size

            # Slicing the matrix to get the current sub-matrix
            chunk_similarities = similarities[:, row_start:row_end, col_start:col_end]

            # For long list, there is no distinction
            query_query_similarity = chunk_similarities
            doc_query_similarity = chunk_similarities
            doc_doc_similarity = chunk_similarities

            (
                local_shortlist_indices,
                local_confidences,
            ) = self.pipeline.subset_selection_strategy.subset_select_with_similarity(
                query_query_similarity, doc_query_similarity, doc_doc_similarity
            )

            # Adjusting local indices to global indices
            global_indices = [
                (idx + col_start).item() for idx in local_shortlist_indices
            ]
            local_confidences = local_confidences.tolist()

            # Combine and keep only top entries
            for idx, conf in zip(global_indices, local_confidences):
                top_indices.append(idx)
                top_confidences.append(conf)

            # If we exceed budget, trim to top-budget items based on confidence
            if len(top_indices) > self.annotation_budget:
                # Sorting by confidence and selecting top-budget entries
                sorted_indices = sorted(
                    zip(top_indices, top_confidences),
                    key=lambda x: x[1],
                    reverse=True,
                )[: self.annotation_budget]
                top_indices, top_confidences = zip(*sorted_indices)
                top_indices = list(top_indices)
                top_confidences = list(top_confidences)

        # Redefining top_indices for the final run
        top_indices = torch.tensor(top_indices)
        top_similarity_matrix = similarities[:, top_indices[:, None], top_indices]

        (
            final_indices,
            final_confidences,
        ) = self.pipeline.subset_selection_strategy.subset_select_with_similarity(
            top_similarity_matrix, top_similarity_matrix, top_similarity_matrix
        )

        # Convert final local indices to the global context
        final_global_indices = top_indices[final_indices].tolist()

        return final_global_indices, final_confidences.tolist()
