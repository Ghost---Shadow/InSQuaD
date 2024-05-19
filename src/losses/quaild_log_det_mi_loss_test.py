import unittest
from losses.utils import bind_pick_most_diverse
import torch
from torch.optim import AdamW
from losses.quaild_log_det_mi_loss import QuaidLogDetMILoss
from config import Config
from train_utils import set_seed
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


# python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss -v
class TestQuaidLogDetMILoss(unittest.TestCase):
    def setUp(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.loss.lambd = 1.0
        self.loss_fn = QuaidLogDetMILoss(config)

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_log_det_happy -v
    def test_log_det_happy(self):
        # Create a tensor representing positive infinity
        matrix = torch.tensor(
            [[4.0, 1.0], [1.0, 3.0]],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda:0",
        )

        # Apply the finitify function
        actual_loss = self.loss_fn.safe_logdet(matrix)
        expected_loss = 2.397895336151123
        self.assertAlmostEqual(actual_loss.item(), expected_loss, places=3)

        # Backward pass to compute gradients
        loss = torch.exp(actual_loss)
        loss.backward()

        assert matrix.grad.tolist() == [
            [3.000000238418579, -1.0000001192092896],
            [-1.0000001192092896, 4.000000476837158],
        ], matrix.grad.tolist()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_log_det_singular -v
    def test_log_det_singular(self):
        # Create a tensor representing positive infinity
        matrix = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda:0",
        )

        # Apply the finitify function
        actual_loss = self.loss_fn.safe_logdet(matrix)
        expected_loss = -6.214068412780762
        self.assertAlmostEqual(actual_loss.item(), expected_loss, places=3)

        # Backward pass to compute gradients
        loss = torch.exp(actual_loss)
        loss.backward()

        assert matrix.grad.tolist() == [
            [1.0010002851486206, -1.000000238418579],
            [-1.000000238418579, 1.0010002851486206],
        ], matrix.grad.tolist()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_safe_pinverse_happy -v
    def test_safe_pinverse_happy(self):
        # Create a tensor representing positive infinity
        matrix = torch.tensor(
            [[4.0, 1.0], [1.0, 3.0]],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda:0",
        )

        # Apply the finitify function
        actual = self.loss_fn.safe_pinverse(matrix)
        assert actual.tolist() == [
            [0.27264466881752014, -0.09085128456354141],
            [-0.09085127711296082, 0.36349597573280334],
        ], actual.tolist()

        # Backward pass to compute gradients
        loss = self.loss_fn.safe_logdet(actual)
        loss = torch.exp(loss)
        assert loss.item() == 0.09148841351270676, loss.item()
        loss.backward()

        assert matrix.grad.tolist() == [
            [-0.024852704256772995, 0.008311750367283821],
            [0.008311749435961246, -0.03316446766257286],
        ], matrix.grad.tolist()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_safe_pinverse_singular -v
    def test_safe_pinverse_singular(self):
        # Create a tensor representing positive infinity
        matrix = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda:0",
        )

        # Apply the finitify function
        actual = self.loss_fn.safe_pinverse(matrix)
        assert actual.tolist() == [
            [500.2230529785156, -499.7232971191406],
            [-499.7232971191406, 500.2230529785156],
        ], actual.tolist()

        # Backward pass to compute gradients
        loss = self.loss_fn.safe_logdet(actual)
        loss = torch.exp(loss)
        assert loss.item() == 500.7322692871094, loss.item()
        loss.backward()

        assert matrix.grad.tolist() == [
            [-250448.109375, 250228.875],
            [250198.359375, -250478.625],
        ], matrix.grad.tolist()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_theoretical_lower_bound -v
    def test_theoretical_lower_bound(self):
        # Construct vectors that should ideally minimize mutual information
        original_a = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True, device="cuda:0"
        )
        original_b = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True, device="cuda:0"
        )

        # Normalize to ensure no scale issues
        a = F.normalize(original_a, p=2, dim=-1)
        b = F.normalize(original_b, p=2, dim=-1)

        # Compute the similarity
        similarity = self.loss_fn.similarity(a, b)
        self.assertAlmostEqual(similarity.item(), 1.0, places=3)

        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

        # Should not crash
        with torch.autograd.detect_anomaly():
            loss = self.loss_fn(a, b)
            loss.backward()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_theoretical_upper_bound -v
    def test_theoretical_upper_bound(self):
        original_a = torch.tensor(
            [[[1.0, 0.0], [-1.0, 0.0]]], requires_grad=True, device="cuda:0"
        )
        original_b = torch.tensor(
            [[[0.0, 1.0], [0.0, -1.0]]], requires_grad=True, device="cuda:0"
        )

        # Normalize to avoid any scale issues
        a = F.normalize(original_a, p=2, dim=-1)
        b = F.normalize(original_b, p=2, dim=-1)

        # Compute the similarity
        similarity = self.loss_fn.similarity(a, b)
        self.assertAlmostEqual(similarity.item(), 0.0, places=3)

        # The expected loss should be close to 0 if mutual information is maximized
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 1.0, places=3)

        # Should not crash
        with torch.autograd.detect_anomaly():
            loss = self.loss_fn(a, b)
            loss.backward()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_dimension_mismatch -v
    def test_dimension_mismatch(self):
        a = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
            requires_grad=True,
            device="cuda:0",
        )
        b = torch.tensor(
            [[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]], requires_grad=True, device="cuda:0"
        )
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 1.0, places=3)

        # Should not crash
        with torch.autograd.detect_anomaly():
            loss = self.loss_fn(a, b)
            loss.backward()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_submodularity -v
    def test_submodularity(self):
        # q = [0.7071, 0.7071, 0.0000] # query
        a = [1.0000, 0.0000, 0.0000]  # 0 # partial match
        b = [0.7071, 0.7071, 0.0000]  # 1 # perflect quality, first pick
        c = [0.0000, 0.0000, 1.0000]  # 2 # completely orthogonal
        d = [0.7071, 0.7071, 0.0000]  # 3 # perfect quality, wrong diversity
        e = [-0.7071, -0.7071, 0.0000]  # 4 # anti-parallel quality

        pick_most_diverse = bind_pick_most_diverse(self)

        best_diversity, best_candidate = pick_most_diverse([b], [a, c, d, e])

        self.assertAlmostEqual(best_diversity, 1.0, places=3)
        assert best_candidate == c, best_candidate

        best_diversity, best_candidate = pick_most_diverse([b, c], [a, e, d])

        self.assertAlmostEqual(best_diversity, 1.0, places=3)
        assert best_candidate == a, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c], [d, e])

        self.assertAlmostEqual(best_diversity, 0.4999200701713562, places=3)
        assert best_candidate == d, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c, d], [e])

        self.assertAlmostEqual(best_diversity, 1.0, places=3)
        assert best_candidate == e, best_candidate

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_submodularity_with_arbitary_order -v
    def test_submodularity_with_arbitary_order(self):
        # q = [0.7071, 0.7071, 0.0000] # query
        a = [1.0000, 0.0000, 0.0000]  # 0 # partial match
        b = [0.7071, 0.7071, 0.0000]  # 1 # perflect quality, first pick
        c = [0.0000, 0.0000, 1.0000]  # 2 # completely orthogonal
        d = [0.7071, 0.7071, 0.0000]  # 3 # perfect quality, wrong diversity
        e = [-0.7071, -0.7071, 0.0000]  # 4 # anti-parallel quality

        # [1, 0, 2, 3, 4]

        pick_most_diverse = bind_pick_most_diverse(self)

        best_diversity, best_candidate = pick_most_diverse([b], [a, c, d, e])

        self.assertAlmostEqual(best_diversity, 1.0, places=3)
        assert best_candidate == c, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b], [c, d, e])

        self.assertAlmostEqual(best_diversity, 1.0, places=3)
        assert best_candidate == c, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c], [d, e])

        self.assertAlmostEqual(best_diversity, 0.4999200701713562, places=3)
        assert best_candidate == d, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c, d], [e])

        self.assertAlmostEqual(best_diversity, 1.0, places=3)
        assert best_candidate == e, best_candidate

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_overfit -v
    def test_overfit(self):
        set_seed(42)

        # embedding_size = 256
        # num_docs = 10
        # batch_size = 2

        # Create random tensors for a and b
        # original_a = torch.randn(
        #     batch_size, num_docs, embedding_size, requires_grad=False
        # )
        # original_b = torch.randn(
        #     batch_size, num_docs, embedding_size, requires_grad=False
        # )
        original_a = torch.tensor(
            [
                [[1.0, 0.01, 0.01], [1.0, 0.01, 0.01]],
                [[0.01, 0.01, 1.0], [0.01, 0.01, -1.0]],
            ],
            device="cuda:0",
        )
        original_b = torch.tensor(
            [
                [[-1.0, 0.01, 0.01], [-1.0, 0.01, 0.01]],
                [[0.01, 0.01, -1.0], [0.01, 0.01, 1.0]],
            ],
            device="cuda:0",
        )
        # original_a = torch.tensor([[[1.0, 0.1, 0.1]]])
        # original_b = torch.tensor([[[-1.0, 0.1, 0.1]]])

        # original_a = torch.tensor([[[1.0, 0.1, 0.1]]])
        # original_b = torch.tensor([[[0.1, 1.0, 0.1]]])

        # Normalize a and b
        normalized_a = F.normalize(original_a, dim=-1)
        normalized_b = F.normalize(original_b, dim=-1)

        # Make a and b require gradients by reassigning them as new tensors
        a = normalized_a.clone().detach().requires_grad_(True)
        b = normalized_b.clone().detach().requires_grad_(True)

        optimizer = AdamW([a, b], lr=1e-0)

        # Training loop
        # for epoch in range(10):
        for epoch in range(1000):
            # print("-" * 80)
            optimizer.zero_grad()
            loss = self.loss_fn(a, b)

            # print(loss)
            if torch.isnan(loss):
                raise Exception("Got nan, stopping")

            mse_loss = torch.nn.functional.mse_loss(a, b, reduction="mean")

            loss.backward()

            # Print gradients for inspection
            # print(f"Gradients of a at Epoch {epoch+1}: {a.grad.tolist()}")
            # print(f"Gradients of b at Epoch {epoch+1}: {b.grad.tolist()}")
            grad_norm_a = torch.norm(a.grad).item()
            grad_norm_b = torch.norm(b.grad).item()

            # if grad_norm_b == 0.0:
            #     raise Exception("B died, stopping")

            optimizer.step()

            # Re-normalize a and b after the update step
            with torch.no_grad():
                a.copy_(F.normalize(a, dim=-1))  # In-place update of 'a'
                b.copy_(F.normalize(b, dim=-1))  # In-place update of 'b'

            print(
                f"Epoch {epoch+1}, Loss: {loss.item()}, MSE: {mse_loss.item()}, ga: {grad_norm_a}, gb: {grad_norm_b}",
                # a.tolist(),
                # b.tolist(),
            )

        # assert mse_loss.item() <= 1.5, mse_loss.item()

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_overfit_amp -v
    def test_overfit_amp(self):
        set_seed(42)

        original_a = torch.tensor(
            [
                [[1.0, 0.01, 0.01], [1.0, 0.01, 0.01]],
                [[0.01, 0.01, 1.0], [0.01, 0.01, -1.0]],
            ],
            device="cuda:0",
        )
        original_b = torch.tensor(
            [
                [[-1.0, 0.01, 0.01], [-1.0, 0.01, 0.01]],
                [[0.01, 0.01, -1.0], [0.01, 0.01, 1.0]],
            ],
            device="cuda:0",
        )

        # Normalize a and b
        normalized_a = F.normalize(original_a, dim=-1)
        normalized_b = F.normalize(original_b, dim=-1)

        # Make a and b require gradients
        a = normalized_a.clone().detach().requires_grad_(True)
        b = normalized_b.clone().detach().requires_grad_(True)

        optimizer = AdamW([a, b], lr=1e-3)
        scaler = GradScaler()  # Initialize GradScaler for AMP

        # Training loop
        # for epoch in range(10):
        for epoch in range(1000):
            # print("-" * 80)
            # for epoch in range(1000):
            optimizer.zero_grad()

            # Forward pass using autocast for mixed precision
            with autocast():
                loss = self.loss_fn(a, b)
                mse_loss = torch.nn.functional.mse_loss(a, b, reduction="mean")

            if torch.isnan(loss):
                raise Exception("Got nan, stopping")

            # Backward pass with scaled loss
            scaler.scale(loss).backward()

            grad_norm_a = torch.norm(a.grad).item()
            grad_norm_b = torch.norm(b.grad).item()

            # Update the parameters
            scaler.step(optimizer)
            scaler.update()

            # Re-normalize a and b after the update step
            with torch.no_grad():
                a.copy_(F.normalize(a, dim=-1))
                b.copy_(F.normalize(b, dim=-1))

            print(
                f"Epoch {epoch+1}, Loss: {loss.item()}, MSE: {mse_loss.item()}, ga: {grad_norm_a}, gb: {grad_norm_b}",
                # a.tolist(),
                # b.tolist(),
            )

        # assert mse_loss.item() <= 1.5, f"MSE Loss is too high: {mse_loss.item()}"


if __name__ == "__main__":
    unittest.main()
