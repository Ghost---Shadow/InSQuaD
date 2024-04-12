from losses.base_loss import BaseLoss
import torch.nn.functional as F


class KLDivLoss(BaseLoss):
    def __init__(self, config=None):
        super(KLDivLoss, self).__init__()
        self.config = config

    def forward(self, input, target):
        # Ensure input and target are probability distributions (e.g., using softmax)
        input_prob = F.log_softmax(input, dim=-1)
        target_prob = F.softmax(target, dim=-1)

        # Compute the KL divergence loss
        return F.kl_div(input_prob, target_prob, reduction="batchmean")
