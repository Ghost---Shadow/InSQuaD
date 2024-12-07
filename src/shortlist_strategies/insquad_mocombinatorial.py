from shortlist_strategies.insquad_combinatorial import InsquadCombinatorialStrategy
from tqdm import tqdm
from config import RootConfig


class InsquadMemoryOptimizedCombinatorialStrategy(InsquadCombinatorialStrategy):
    NAME = "insquad_mocombinatorial"

    def __init__(self, config: RootConfig, pipeline):
        super().__init__(config, pipeline)

        if (
            config.offline_validation.subsample_for_train_size
            % config.offline_validation.insquad_mocombinatorial_config.window_size
            != 0
        ):
            raise ValueError(
                "offline_validation.insquad_mocombinatorial_config.window_size is not divisible by offline_validation.subsample_for_train_size"
            )

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
