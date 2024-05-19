from losses.base_loss import BaseLoss
import torch.nn as nn


class MSELoss(BaseLoss):
    NAME = "mean_squared_error"

    def __init__(self, config=None):
        super(MSELoss, self).__init__(config)

    def forward(self, input, target):
        # Compute the MSE loss
        return nn.functional.mse_loss(input, target, reduction="mean")
