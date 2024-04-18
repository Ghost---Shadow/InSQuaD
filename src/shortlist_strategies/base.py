from abc import abstractmethod
from config import RootConfig


class BaseStrategy:
    def __init__(self, config: RootConfig, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.top_n = self.config.offline_validation.annotation_budget

    def subsample_dataset(self, wrapped_dataset, split):
        # TODO: Shuffle the subsample
        split = wrapped_dataset.split_lut[split]
        dataset_length = len(wrapped_dataset.dataset[split])
        row_iterator = wrapped_dataset.get_row_iterator(split)
        subsample_for_eval_size = self.config.offline_validation.subsample_for_eval_size
        if subsample_for_eval_size is None:
            subsample_for_eval_size = dataset_length

        total = min(dataset_length, subsample_for_eval_size)

        def _iterator():
            i = 0
            for row in row_iterator:
                yield row
                i += 1
                if i == total:
                    break

        return total, _iterator()

    def _populate_and_cache_index(self, cache_name, use_cache, wrapped_dataset):
        if use_cache and self.pipeline.dense_index.does_cache_exist(cache_name):
            print(f"Found dense index cache {cache_name}")
            self.pipeline.dense_index.load_index(wrapped_dataset, cache_name)
        else:
            # Populate dense index
            self.pipeline.dense_index.repopulate_index(
                wrapped_dataset,
                self.pipeline.semantic_search_model,
            )
            self.pipeline.dense_index.save_index(cache_name)

    @abstractmethod
    def shortlist(self, dataset_name, use_cache=True): ...

    @abstractmethod
    def assemble_few_shot(self, dataset_name, use_cache=True): ...
