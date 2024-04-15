from config import RootConfig
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLinearScheduler(_LRScheduler):
    NAME = "linear"

    def __init__(self, config: RootConfig, optimizer, train_loader, last_step):
        total_epochs = config.training.epochs
        warmup_ratio = config.training.warmup_ratio

        (
            self.num_warmup_steps,
            self.num_training_steps,
        ) = WarmupLinearScheduler.compute_scheduler_steps(
            train_loader, total_epochs, warmup_ratio
        )

        # epoch = step, Bad nomenclature
        super(WarmupLinearScheduler, self).__init__(optimizer, last_epoch=last_step)

    @staticmethod
    def compute_scheduler_steps(train_loader, total_epochs, warmup_ratio):
        num_batches_per_epoch = len(train_loader.dataset["train"])
        num_training_steps = num_batches_per_epoch * total_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        return num_warmup_steps, num_training_steps

    def get_lr(self):
        if self._step_count < self.num_warmup_steps:
            # Scale up from 0 to the base learning rate
            return [
                base_lr * float(self._step_count) / float(max(1, self.num_warmup_steps))
                for base_lr in self.base_lrs
            ]
        # Scale down from base learning rate to 0
        return [
            base_lr
            * max(
                0.0,
                (
                    float(self.num_training_steps - self._step_count)
                    / float(max(1, self.num_training_steps - self.num_warmup_steps))
                ),
            )
            for base_lr in self.base_lrs
        ]
