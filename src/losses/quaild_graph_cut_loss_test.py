import unittest
import torch

from losses.quaild_graph_cut_loss import QuaildGraphCutLoss
from config import Config


class TestQuaildGraphCut(unittest.TestCase):
    def setUp(self):
        # Setup code runs before each test method
        self.config = Config.from_file("experiments/quaild_test_experiment.yaml")

        self.model = QuaildGraphCutLoss(self.config)
        self.a = torch.rand((5, 3))  # Example tensor a
        self.b = torch.rand((5, 3))  # Example tensor b

    def test_forward_output_shape(self):
        """Test if the output of the forward method has the correct shape."""
        loss = self.model(self.a, self.b)
        self.assertIsInstance(loss, torch.Tensor)  # Check if output is a torch tensor
        self.assertEqual(loss.shape, torch.Size([]))  # Check if output is a scalar

    def test_forward_positive_loss(self):
        """Test if the loss is positive."""
        loss = self.model(self.a, self.b)
        self.assertGreater(loss.item(), 0)  # Check if loss is greater than 0

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


if __name__ == "__main__":
    unittest.main()
