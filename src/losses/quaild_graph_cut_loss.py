import torch
import torch.nn as nn


class QuaildGraphCutLoss(nn.Module):
    def __init__(self, config):
        super(QuaildGraphCutLoss, self).__init__()
        self.lambd = config.training.loss.lambd
        self.epsilon = 0.0  # Adjust if necessary

    def forward(self, a, b):
        if a.shape == 2:
            a = a.unsqueeze(0)
        if b.shape == 2:
            b = b.unsqueeze(0)
        assert len(a.shape) == 3
        assert len(b.shape) == 3
        batch_size, num_docs, features = a.shape

        # Ensure a and b are 2D [batch_size, num_docs, features]
        b_t = b.transpose(1, 2)  # Now [batch_size, features, num_docs]

        # Matrix multiplication [batch_size, num_docs, num_docs]
        similarity = torch.matmul(a, b_t)

        # Normalize and rescale
        similarity = similarity.reshape([batch_size, num_docs * num_docs])
        aggregated_similarity = 2 * self.lambd * similarity.mean(dim=-1)

        loss = -aggregated_similarity

        # Adjust the lower bound for each item in the batch
        theoretical_lower_bound = 2 * self.lambd * -1
        loss = loss - theoretical_lower_bound + self.epsilon

        return loss.mean()
