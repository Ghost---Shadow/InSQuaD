from learning_rate_schedulers.no_operation import NoOpLRScheduler
from learning_rate_schedulers.warmup_linear_scheduler import WarmupLinearScheduler


LEARNING_RATE_SCHEDULERS_LUT = {
    NoOpLRScheduler.NAME: NoOpLRScheduler,
    WarmupLinearScheduler.NAME: WarmupLinearScheduler,
}
