import unittest
import torch

from losses.quaild_facility_location_loss import QuaildFacilityLocationLoss
from config import Config


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


if __name__ == "__main__":
    unittest.main()
