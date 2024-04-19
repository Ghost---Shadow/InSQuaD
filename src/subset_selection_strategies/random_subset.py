import numpy as np


class RandomSubsetStrategy:
    NAME = "random"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.budget = self.config.offline_validation.annotation_budget

    def subset_select(self, embeddings):
        total_dataset_length = len(embeddings)
        indexes = np.arange(len(embeddings))
        np.random.shuffle(indexes)
        indexes = indexes[: self.config.offline_validation.annotation_budget]
        indexes = indexes.tolist()
        confidences = [1 / total_dataset_length] * len(indexes)  # Uniform
        return indexes, confidences
