import unittest
import torch
from torch.optim import AdamW
from losses.quaild_graph_cut_loss import QuaildGraphCutLoss
from config import Config
from train_utils import set_seed


# python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut -v
class TestQuaildGraphCut(unittest.TestCase):
    def setUp(self):
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        self.loss_fn = QuaildGraphCutLoss(config)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_theoretical_lower_bound -v
    def test_theoretical_lower_bound(self):
        a = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        b = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 0.0)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_theoretical_upper_bound -v
    def test_theoretical_upper_bound(self):
        a = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        b = torch.tensor(
            [[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]]
        )
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 2.0)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_dimension_mismatch -v
    def test_dimension_mismatch(self):
        a = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        b = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 0.0)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_overfit -v
    def test_overfit(self):
        set_seed(42)

        # embedding_size = 256
        # num_docs = 10
        # batch_size = 2

        embedding_size = 3
        num_docs = 4
        batch_size = 2

        # Create random tensors for a and b
        original_a = torch.randn(
            batch_size, num_docs, embedding_size, requires_grad=False
        )
        original_b = torch.randn(
            batch_size, num_docs, embedding_size, requires_grad=False
        )

        # Normalize a and b
        normalized_a = original_a / torch.norm(original_a)
        normalized_b = original_b / torch.norm(original_b)

        # Make a and b require gradients by reassigning them as new tensors
        a = torch.tensor(normalized_a, requires_grad=True)
        b = torch.tensor(normalized_b, requires_grad=True)

        optimizer = AdamW([a, b], lr=0.1)

        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()
            loss = self.loss_fn(a, b)

            mse_loss = torch.nn.functional.mse_loss(a, b, reduction="mean")

            loss.backward()
            optimizer.step()

            # Re-normalize a and b after the update step
            with torch.no_grad():
                a /= torch.norm(a)
                b /= torch.norm(b)

            # print(
            #     f"Epoch {epoch+1}, Loss: {loss.item()}, MSE: {mse_loss.item()}",
            #     # a.tolist(),
            #     # b.tolist(),
            # )

        assert mse_loss.item() <= 1e-4, mse_loss.item()


if __name__ == "__main__":
    unittest.main()
