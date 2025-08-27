from abc import abstractmethod
import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self, config):
        super(BaseLoss, self).__init__()
        self.config = config

    def ensure_3d(self, tensor):
        """
        Ensure the input tensor has three dimensions [batch_size, num_vectors, vector_dimension].
        If the tensor is 2D, interpret it as [1, num_vectors, vector_dimension].
        """
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        assert len(tensor.shape) == 3, tensor.shape
        return tensor

    def similarity(self, a, b):
        # Ensure a and b are float tensors (necessary for torch.cosine_similarity)
        a = a.float()
        b = b.float()

        if len(a.shape) == 3 and len(b.shape) == 3:
            # If a and b are 3-dimensional: [batch_size, num_docs, embedding_dim]
            # Normalize along the embedding dimension
            a_norm = F.normalize(a, p=2, dim=2)
            b_norm = F.normalize(b, p=2, dim=2)
            # Calculate the cosine similarity along the embedding_dim, resulting in a [batch_size, num_docs, num_docs] tensor
            cos_sim = torch.bmm(a_norm, b_norm.transpose(1, 2))
        elif len(a.shape) == 2 and len(b.shape) == 2:
            # If a and b are 2-dimensional: [batch_size, embedding_dim]
            # Normalize along the embedding dimension
            a_norm = F.normalize(a, p=2, dim=1)
            b_norm = F.normalize(b, p=2, dim=1)
            # Calculate the cosine similarity, resulting in a [batch_size] tensor when a and b are batches of vectors
            cos_sim = torch.sum(a_norm * b_norm, dim=1)
        else:
            raise ValueError("Unsupported tensor shapes.")

        return cos_sim.mean()

    def compute_similarity_matrix(self, a, b, center_at_half=True):
        a = self.ensure_3d(a)
        b = self.ensure_3d(b)

        assert all(dim > 0 for dim in a.shape), "a has zero-size dimension"
        assert all(dim > 0 for dim in b.shape), "b has zero-size dimension"

        # Compute similarity matrix S using dot product
        # S_ij = a_i . b_j^T
        # [batch_size, num_docs, features] to [batch_size, num_docs, num_docs]
        similarities = torch.matmul(a, b.transpose(1, 2))
        if center_at_half:
            similarities = (similarities + 1.0) / 2.0  # Center at 0.5
        return similarities

    @abstractmethod
    def forward(self, a, b):
        ...
