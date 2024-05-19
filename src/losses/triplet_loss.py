from losses.base_loss import BaseLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(BaseLoss):
    def __init__(self, config):
        super(TripletLoss, self).__init__(config)
        self.config = config
        self.margin = 1.0  # TODO

    def forward(self, similarity, labels):
        """
        Forward pass for the triplet loss.
        Args:
            similarity (torch.Tensor): Tensor containing dot product similarities
                                       between question and answer embeddings.
            labels (torch.Tensor): Float32 Tensor where 1 = positive, 0 = negative

        Returns:
            torch.Tensor: The computed triplet loss.
        """
        assert len(similarity.shape) == 1, "TODO: Batch support"
        assert len(labels.shape) == 1, "TODO: Batch support"

        # Convert float32 to bool mask
        labels = labels > 0

        # If all negative, just dont crash
        if labels.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        # if all positive, just dont crash
        if labels.sum() == labels.shape[-1]:
            return torch.tensor(0.0, requires_grad=True)

        # Extract positive and negative similarities
        positive_similarities = similarity[labels]
        negative_similarities = similarity[~labels]

        # Calculate triplet loss
        # The loss is max(0, margin + negative_sim - positive_sim) for each positive/negative pair
        loss = 0.0
        for pos_sim in positive_similarities:
            # Calculate the loss for each positive example against all
            # negative examples
            losses = F.relu(self.margin + negative_similarities - pos_sim)
            # Average loss for the current positive example
            loss += losses.mean()

        # Average over all positive examples
        return loss / len(positive_similarities)
