import unittest
from learning_rate_schedulers.warmup_linear_scheduler import WarmupLinearScheduler
import torch
import matplotlib.pyplot as plt
from dataloaders.hotpot_qa_with_q_loader import get_train_loader


# python -m unittest learning_rate_schedulers.warmup_linear_scheduler_test.TestWarmupLinearScheduler -v
class TestWarmupLinearScheduler(unittest.TestCase):
    def test_scheduler(self):
        batch_size = 48
        total_epochs = 10
        warmup_ratio = 0.06
        base_lr = 3e-5
        last_step = -1
        filename = "./artifacts/TestWarmupLinearScheduler.png"

        train_loader = get_train_loader(batch_size)

        config = {
            "training": {
                "epochs": total_epochs,
                "warmup_ratio": warmup_ratio,
            }
        }

        optimizer = torch.optim.Adam([torch.zeros(3, requires_grad=True)], lr=base_lr)
        scheduler = WarmupLinearScheduler(config, optimizer, train_loader, last_step)

        num_steps_per_epoch = len(train_loader)

        lrs = []
        for _ in range(total_epochs):
            for _ in range(num_steps_per_epoch):
                optimizer.step()
                scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])

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
