import unittest
import torch
from losses.triplet_loss import TripletLoss


# python -m unittest losses.triplet_loss_test.TestTripletLoss -v
class TestTripletLoss(unittest.TestCase):
    # python -m unittest losses.triplet_loss_test.TestTripletLoss.test_loss_computation -v
    def test_loss_computation(self):
        input_tensor = torch.tensor([0.9, 0.7, 0.6], requires_grad=True)
        target_tensor = torch.tensor([1, 0, 1])

        expected_loss = torch.tensor([0.9500000476837158])

        config = {}

        loss_fn = TripletLoss(config)

        loss = loss_fn(input_tensor, target_tensor)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        loss.backward()

    # python -m unittest losses.triplet_loss_test.TestTripletLoss.test_loss_computation_all_negative -v
    def test_loss_computation_all_negative(self):
        input_tensor = torch.tensor([0.9, 0.7, 0.6], requires_grad=True)
        target_tensor = torch.tensor([0, 0, 0])

        expected_loss = torch.tensor([0.0])

        config = {}

        loss_fn = TripletLoss(config)

        loss = loss_fn(input_tensor, target_tensor)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        loss.backward()

    # python -m unittest losses.triplet_loss_test.TestTripletLoss.test_loss_computation_all_positive -v
    def test_loss_computation_all_positive(self):
        input_tensor = torch.tensor([0.9, 0.7, 0.6], requires_grad=True)
        target_tensor = torch.tensor([1, 1, 1])

        expected_loss = torch.tensor([0.0])

        config = {}

        loss_fn = TripletLoss(config)

        loss = loss_fn(input_tensor, target_tensor)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        loss.backward()


if __name__ == "__main__":
    unittest.main()
