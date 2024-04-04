import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, config=None):
        super(MSELoss, self).__init__()
        self.config = config

    def forward(self, input, target):
        # Compute the MSE loss
        return nn.functional.mse_loss(input, target, reduction="mean")
