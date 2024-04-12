import torch
from semantic_search_models.base import WrappedBaseModel
from sentence_transformers.util import batch_to_device


class WrappedMpnetModel(WrappedBaseModel):
    """
    https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    """

    def __init__(self, config):
        super().__init__(config)

    def embed(self, sentences):
        features = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        features = batch_to_device(features, self.device)
        model_output = self.model(**features)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings

    def compute_quality_diversity(self, question_embeddings, retrieved_documents):
        quality_vectors = []
        diversity_matrices = []

        for question_embedding, doc_batch in zip(
            question_embeddings, retrieved_documents
        ):
            prompts = doc_batch["prompts"]
            document_embeddings = self.embed(prompts)

            # document_embeddings.shape = [num_docs, embedding_size]
            # question_embedding.shape = [embedding_size]
            question_embedding_2d = question_embedding.unsqueeze(0)

            # Compute inter-document similarity
            inter_doc_similarity = torch.mm(
                document_embeddings, document_embeddings.t()
            )
            diversity_matrix = 1 - inter_doc_similarity
            diversity_matrices.append(diversity_matrix)

            # Compute quality vector
            quality_vector = torch.mm(
                document_embeddings, question_embedding_2d.t()
            ).flatten()
            quality_vectors.append(quality_vector)

        # Stack the list of tensors to form a single tensor for each.
        quality_vectors = torch.stack(quality_vectors)
        diversity_matrices = torch.stack(diversity_matrices)

        # quality_vectors.shape = [batch_size, num_docs]
        # diversity_matrices.shape = [batch_size, num_docs, num_docs]
        return quality_vectors, diversity_matrices
