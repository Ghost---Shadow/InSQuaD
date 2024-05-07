from abc import abstractmethod
import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseLoss(nn.Module):
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

    @abstractmethod
    def forward(self, a, b): ...
