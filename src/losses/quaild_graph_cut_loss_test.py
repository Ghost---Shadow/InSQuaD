import unittest
import torch
from torch.optim import AdamW
from losses.quaild_graph_cut_loss import QuaildGraphCutLoss
from config import Config
from train_utils import set_seed


# python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut -v
class TestQuaildGraphCut(unittest.TestCase):
    def setUp(self):
        # Setup code runs before each test method
        self.config = Config.from_file("experiments/quaild_test_experiment.yaml")

        self.model = QuaildGraphCutLoss(self.config)
        self.a = torch.rand((5, 3))  # Example tensor a
        self.b = torch.rand((5, 3))  # Example tensor b

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_forward_output_shape -v
    def test_forward_output_shape(self):
        """Test if the output of the forward method has the correct shape."""
        loss = self.model(self.a, self.b)
        self.assertIsInstance(loss, torch.Tensor)  # Check if output is a torch tensor
        self.assertEqual(loss.shape, torch.Size([]))  # Check if output is a scalar

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_forward_positive_loss -v
    def test_forward_positive_loss(self):
        """Test if the loss is positive."""
        loss = self.model(self.a, self.b)
        self.assertGreater(loss.item(), 0)  # Check if loss is greater than 0

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_theoretical_lower_bound -v
    def test_theoretical_lower_bound(self):
        a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        b = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        loss = self.model(a, b)
        self.assertAlmostEqual(loss.item(), 0.0)
        print(loss)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_lambd_effect -v
    def test_lambd_effect(self):
        """Test if changing lambd affects the loss."""
        initial_loss = self.model(self.a, self.b)
        # Change lambda value and re-test
        self.config.training.loss.lambd = 1.0  # Change lambda to a different value
        # Reinitialize the model with the new lambda
        self.model = QuaildGraphCutLoss(self.config)
        new_loss = self.model(self.a, self.b)
        # Check if loss values are different
        self.assertNotEqual(initial_loss.item(), new_loss.item())

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_overfit -v
    def test_overfit(self):
        set_seed(42)

        # embedding_size = 256
        # batch_size = 10

        embedding_size = 3
        batch_size = 10

        # Create random tensors for a and b
        original_a = torch.randn(batch_size, embedding_size, requires_grad=False)
        original_b = torch.randn(batch_size, embedding_size, requires_grad=False)

        # Normalize a and b
        normalized_a = original_a / torch.norm(original_a)
        normalized_b = original_b / torch.norm(original_b)

        # Make a and b require gradients by reassigning them as new tensors
        a = torch.tensor(normalized_a, requires_grad=True)
        b = torch.tensor(normalized_b, requires_grad=True)

        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        loss_fn = QuaildGraphCutLoss(config)
        optimizer = AdamW([a, b], lr=0.01)

        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()
            loss = loss_fn(a, b)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            # Re-normalize a and b after the update step
            with torch.no_grad():
                a /= torch.norm(a)
                b /= torch.norm(b)

            cosine = torch.mean(torch.matmul(a, b.transpose(0, 1)))
            print(
                f"Epoch {epoch+1}, Loss: {loss.item()}, Cosine: {cosine.item()}",
                # a.tolist(),
                # b.tolist(),
            )

        assert cosine.item() <= 0.0, cosine.item()


if __name__ == "__main__":
    unittest.main()
