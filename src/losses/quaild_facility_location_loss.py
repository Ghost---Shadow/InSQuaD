import torch
import torch.nn as nn


class QuaildFacilityLocationLoss(nn.Module):
    def __init__(self, config):
        """
        Initialize the FacilityLocationLoss module.

        Parameters:
        - lambd (float): Regularization parameter.
        """
        super(QuaildFacilityLocationLoss, self).__init__()
        self.lambd = config.training.loss.lambd

    def forward(self, a, b):
        """
        Compute the Facility Location loss between two sets of embeddings.

        Parameters:
        - a (Tensor): The embeddings tensor of shape (batch_size, embedding_size).
        - b (Tensor): The embeddings tensor of shape (batch_size, embedding_size),
                      representing a different set or the same set as 'a'.

        Returns:
        - loss (Tensor): The computed Facility Location loss.
        """
        # Compute similarity matrix S using dot product
        # S_ij = a_i . b_j^T
        similarities = torch.matmul(a, b.transpose(0, 1))

        # Compute max similarity for each i in a with all j in b
        max_similarities_a_to_b, _ = torch.max(similarities, dim=1)
        max_similarities_b_to_a, _ = torch.max(similarities, dim=0)

        # Sum these max similarities and apply the regularization term
        loss = (
            max_similarities_a_to_b.sum() + self.lambd * max_similarities_b_to_a.sum()
        )

        return loss
