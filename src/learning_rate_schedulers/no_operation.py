from config import RootConfig
from torch.optim.lr_scheduler import _LRScheduler


class NoOpLRScheduler(_LRScheduler):
    NAME = "noop"

    def __init__(self, config: RootConfig, optimizer, wrapped_dataset, last_step):
        # Initialize with the parent class
        super(NoOpLRScheduler, self).__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        # Return the initial learning rates for all parameter groups
        return [base_lr for base_lr in self.base_lrs]
