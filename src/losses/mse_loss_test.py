import unittest
import torch
from losses.mse_loss import MSELoss


# python -m unittest losses._mse_loss_test.TestMSELoss -v
class TestMSELoss(unittest.TestCase):
    # python -m unittest losses.mse_loss_test.TestMSELoss.test_loss_computation -v
    def test_loss_computation(self):
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True
        )
        target_tensor = torch.tensor([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]])

        expected_loss = torch.tensor([0.04666666314005852])
        loss_fn = MSELoss()

        loss = loss_fn(input_tensor, target_tensor)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        # Should not crash
        loss.backward()


if __name__ == "__main__":
    unittest.main()
