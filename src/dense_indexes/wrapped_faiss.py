from dataloaders.base import BaseDataset
import faiss
import torch
from tqdm import tqdm


class WrappedFaiss:
    def __init__(self, config):
        self.config = config
        self.k = config.architecture.dense_index.k_for_rerank
        self.index_class = config.architecture.dense_index.index_class  # IndexFlatL2
        self.wrapped_dataset = None
        self.index = None

    def repopulate_index(self, wrapped_dataset: BaseDataset, embedding_model):
        if self.wrapped_dataset is None:
            self.wrapped_dataset = wrapped_dataset

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
        for row in tqdm(
            self.wrapped_dataset.dataset["train"], desc="populating index", leave=False
        ):
            batch = self.wrapped_dataset.collate_fn([row])
            for prompt in batch["prompts"]:
                with torch.no_grad():
                    embedding = embedding_model.embed(prompt).cpu().numpy()
                self.index.add(embedding)

    def retrieve(self, query_embeddings: torch.Tensor):
        assert self.index.is_trained

        k = self.k + 1  # Adjusting k to account for self-match
        query_embeddings = query_embeddings.detach().cpu().numpy()
        distances, indices = self.index.search(query_embeddings, k)

        batch_results = []
        for query_idx in range(distances.shape[0]):
            # Exclude the most similar result (assuming it's self) for each query
            query_distances = distances[query_idx, 1:]
            query_indices = indices[query_idx, 1:]

            batch = self.wrapped_dataset.random_access(query_indices)
            batch["distances"] = []

            for distance in query_distances:
                batch["distances"].append(float(distance))

            batch_results.append(batch)

        return batch_results
