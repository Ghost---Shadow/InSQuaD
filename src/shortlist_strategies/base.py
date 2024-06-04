from abc import abstractmethod
import json
import numpy as np


class BaseStrategy:
    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.top_n = self.config.offline_validation.annotation_budget
        self.subsampled_train_idxs = None
        self.subsampled_eval_idxs = None

    def subsample_dataset_for_train(self):
        assert (
            self.pipeline.current_dataset_name is not None
        ), "pipeline.current_dataset_name not set"
        # TODO: Cache load

        # Same as baseline papers
        subsample_size = self.config.offline_validation.subsample_for_train_size
        wrapped_dataset = self.pipeline.offline_dataset_lut[
            self.pipeline.current_dataset_name
        ]
        subsample_size = min(wrapped_dataset.get_length("train"), subsample_size)

        self.subsampled_train_idxs, _iterator = self.subsample_dataset(
            wrapped_dataset, "train", subsample_size
        )
        rows = list(_iterator)  # TODO: Optimize
        assert subsample_size == len(rows), len(rows)
        with open(self.pipeline.longlisted_data_path, "w") as f:
            # For inspection only
            json.dump(rows, f, indent=2)

        return rows

    def subsample_dataset_for_eval(self):
        assert (
            self.pipeline.current_dataset_name is not None
        ), "pipeline.current_dataset_name not set"
        # TODO: Cache load

        # Same as baseline papers
        subsample_size = self.config.offline_validation.subsample_for_eval_size
        wrapped_dataset = self.pipeline.offline_dataset_lut[
            self.pipeline.current_dataset_name
        ]
        subsample_size = min(wrapped_dataset.get_length("validation"), subsample_size)
        self.subsampled_eval_idxs, _iterator = self.subsample_dataset(
            wrapped_dataset, "validation", subsample_size
        )
        rows = list(_iterator)  # TODO: Optimize
        assert subsample_size == len(rows), len(rows)
        return rows

    def subsample_dataset(self, wrapped_dataset, split, subsample_size):
        split = wrapped_dataset.split_lut[split]
        dataset_length = len(wrapped_dataset.dataset[split])

        if subsample_size is None:
            subsample_size = dataset_length

        # Don't try to sample larger than population
        subsample_size = min(subsample_size, dataset_length)

        # Use the local generator to sample indices
        assert self.pipeline.current_seed is not None
        rng = np.random.default_rng(seed=self.pipeline.current_seed)
        subsampled_indices = rng.choice(
            dataset_length, subsample_size, replace=False
        ).tolist()

        def _iterator():
            for idx in subsampled_indices:
                yield wrapped_dataset.get_row(split, idx)

        return subsampled_indices, _iterator()

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
    def shortlist(self, use_cache=True): ...

    @abstractmethod
    def assemble_few_shot(self, use_cache=True): ...
