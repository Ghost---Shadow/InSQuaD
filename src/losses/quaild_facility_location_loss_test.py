import unittest
import torch
import torch.nn as nn
from torch.optim import AdamW
from losses.quaild_facility_location_loss import QuaildFacilityLocationLoss
from config import Config
from train_utils import set_seed


# python -m unittest losses.quaild_facility_location_loss_test.TestQuaildFacilityLocationLoss -v
class TestQuaildFacilityLocationLoss(unittest.TestCase):
    def setUp(self):
        self.embedding_size = 4
        self.batch_size = 3
        self.a = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        self.b = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        )

    # python -m unittest losses.quaild_facility_location_loss_test.TestQuaildFacilityLocationLoss.test_loss_value_known_inputs -v
    def test_loss_value_known_inputs(self):
        """Test the loss computation with known inputs."""
        lambd = 1.0
        config = Config.from_file("experiments/quaild_test_experiment.yaml")
        config.training.loss.lambd = lambd
        config.training.loss.type = "facility_location"
        loss_module = QuaildFacilityLocationLoss(config)
        loss = loss_module(self.a, self.b)
        expected_loss_value = 3.0 + 3.0 * lambd
        self.assertAlmostEqual(loss.item(), expected_loss_value, places=4)

    # python -m unittest losses.quaild_facility_location_loss_test.TestQuaildFacilityLocationLoss.test_lambd_impact -v
    def test_lambd_impact(self):
        """Test the impact of lambda on the loss value."""
        config = Config.from_file("experiments/quaild_test_experiment.yaml")

        lambd_values = [0.5, 1.0, 1.5]
        previous_loss = None
        for lambd in lambd_values:
            config.training.loss.lambd = lambd
            loss_module = QuaildFacilityLocationLoss(config)
            loss = loss_module(self.a, self.b)
            if previous_loss is not None:
                # Ensure that increasing lambda increases the loss value.
                self.assertGreater(loss.item(), previous_loss.item())
            previous_loss = loss

    # python -m unittest losses.quaild_facility_location_loss_test.TestQuaildFacilityLocationLoss.test_overfit -v
    def test_overfit(self):
        set_seed(42)

        embedding_size = 3
        batch_size = 1

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
        loss_fn = QuaildFacilityLocationLoss(config)
        optimizer = AdamW([a, b], lr=0.01)

        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()
            loss = loss_fn(a, b)
            loss.backward()
            optimizer.step()

            # Re-normalize a and b after the update step
            with torch.no_grad():
                a /= torch.norm(a)
                b /= torch.norm(b)

            cosine = torch.sum(torch.matmul(a, b.transpose(0, 1)))
            # print(
            #     f"Epoch {epoch+1}, Loss: {loss.item()}, Cosine: {cosine.item()}",
            #     a.tolist(),
            #     b.tolist(),
            # )

        assert cosine.item() < -0.9, cosine.item()


if __name__ == "__main__":
    unittest.main()
