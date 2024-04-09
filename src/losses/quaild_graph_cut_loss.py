import torch
import torch.nn as nn


class QuaildGraphCutLoss(nn.Module):
    def __init__(self, config):
        super(QuaildGraphCutLoss, self).__init__()
        self.lambd = config.training.loss.lambd
        self.epsilon = 0.0  # Adjust if necessary

    def forward(self, a, b):
        # Ensure a and b are 2D [batch_size, features]
        b_t = b.transpose(0, 1)  # Now [features, batch_size]

        # Matrix multiplication [batch_size, batch_size]
        similarity = torch.matmul(a, b_t)

        # Adjust the computation here if you intend to have a different form of aggregation
        # Normalize by number of elements in b
        loss = 2 * self.lambd * similarity.sum(dim=-1)

        # Adjust the lower bound for each item in the batch, if necessary
        theoretical_lower_bound = 2 * self.lambd * -1
        loss = loss - theoretical_lower_bound + self.epsilon

        return loss.mean()
