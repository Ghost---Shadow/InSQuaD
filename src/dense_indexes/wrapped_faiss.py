from pathlib import Path
from dataloaders.base import BaseDataset
import faiss
import torch
from tqdm import tqdm


class WrappedFaiss:
    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.k = config.architecture.dense_index.k_for_rerank
        self.index_class = config.architecture.dense_index.index_class  # IndexFlatL2
        self.wrapped_dataset = None
        self.index = None

    def does_cache_exist(self, cache_name):
        return (self.cache_base_path / cache_name).exists()

    @property
    def cache_base_path(self):
        return Path(self.pipeline.artifacts_dir)

    def save_index(self, cache_name):
        faiss.write_index(self.index, str(self.cache_base_path / cache_name))

    def load_index(self, wrapped_dataset, cache_name):
        # TODO: Single responsibility
        self.wrapped_dataset = wrapped_dataset
        self.index = faiss.read_index(str(self.cache_base_path / cache_name))

    def repopulate_index(self, wrapped_dataset: BaseDataset, embedding_model):
        # TODO: Single responsibility
        self.wrapped_dataset = wrapped_dataset
        assert self.wrapped_dataset is not None

        embedding_dim = embedding_model.model.config.hidden_size
        embedding_model.model.eval()

        # Check and reset/delete the existing index
        if self.index is not None:
            del self.index

        # Create a new index based on index_class from config
        if self.index_class == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unsupported index class: {self.index_class}")

        # Prepare dataset embeddings
        # TODO: Unhardcode
        for row in tqdm(
            self.wrapped_dataset.dataset["train"], desc="populating index", leave=False
        ):
            batch = self.wrapped_dataset.collate_fn([row])
            for prompt in batch["prompts"]:
                with torch.no_grad():
                    embedding = embedding_model.embed(prompt).cpu().numpy()
                self.index.add(embedding)

    def retrieve(self, query_embeddings: torch.Tensor):
        assert self.wrapped_dataset is not None
        assert self.index.is_trained

        k = self.k + 1  # Adjusting k to account for self-match
        query_embeddings = query_embeddings.detach().cpu().numpy()
        distances, indices = self.index.search(query_embeddings, k)

        batch_results = []
        for query_idx in range(distances.shape[0]):
            # Exclude the most similar result (assuming it's self) for each query
            query_distances = distances[query_idx, 1:]
            query_indices = indices[query_idx, 1:].tolist()

            batch = self.wrapped_dataset.random_access(query_indices)
            batch["distances"] = []
            batch["global_indices"] = []

            for global_index, distance in zip(query_indices, query_distances):
                batch["global_indices"].append(int(global_index))
                batch["distances"].append(float(distance))

            batch_results.append(batch)

        return batch_results
