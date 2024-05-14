import unittest
from config import Config
from dataloaders.mrpc import MRPC
from learning_rate_schedulers.warmup_linear_scheduler import WarmupLinearScheduler
import torch
import matplotlib.pyplot as plt


# python -m unittest learning_rate_schedulers.warmup_linear_scheduler_test.TestWarmupLinearScheduler -v
class TestWarmupLinearScheduler(unittest.TestCase):
    def test_scheduler(self):
        last_step = -1
        filename = "./TestWarmupLinearScheduler.png"

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1
        mrpc_dataset = MRPC(config)
        train_loader = mrpc_dataset.get_loader(split="train")
        base_lr = config.training.learning_rate

        optimizer = torch.optim.Adam([torch.zeros(3, requires_grad=True)], lr=base_lr)
        scheduler = WarmupLinearScheduler(config, optimizer, mrpc_dataset, last_step)

        num_steps_per_epoch = len(train_loader)
        print(num_steps_per_epoch)
        print(config.training.epochs)

        lrs = []
        for _ in range(config.training.epochs):
            for _ in range(num_steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])

        # Enable LaTeX typesetting
        # plt.rc("text", usetex=True)
        # plt.rc("font", family="serif")
        plt.figure(figsize=(10, 4))
        plt.plot(lrs, label="Learning Rate")
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule - TestWarmupLinearScheduler")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)

        self.assertAlmostEqual(max(lrs), base_lr)
        self.assertAlmostEqual(min(lrs), 0.0)


if __name__ == "__main__":
    unittest.main()
