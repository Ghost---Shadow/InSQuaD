import unittest
from losses.utils import bind_pick_most_diverse
import torch
from torch.optim import AdamW
from losses.quaild_log_det_mi_loss import QuaidLogDetMILoss
from config import Config
from train_utils import set_seed
import torch.nn.functional as F


# python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss -v
class TestQuaidLogDetMILoss(unittest.TestCase):
    def setUp(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.loss.lambd = 1.0
        self.loss_fn = QuaidLogDetMILoss(config)

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_finitify_inf -v
    def test_finitify_inf(self):
        inf = -torch.log(torch.tensor([0.0]))
        finite_inf = self.loss_fn.finitify(inf)
        positive_inf = self.loss_fn.positive_inf
        self.assertAlmostEqual(finite_inf.item(), positive_inf, places=5)

        neg_inf = -inf
        finite_inf = self.loss_fn.finitify(neg_inf)
        self.assertAlmostEqual(finite_inf.item(), -positive_inf, places=5)

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_finitify_finite -v
    def test_finitify_finite(self):
        finite = torch.tensor([4.2])
        result_finite = self.loss_fn.finitify(finite)
        self.assertAlmostEqual(finite.item(), result_finite.item(), places=5)

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_theoretical_lower_bound -v
    def test_theoretical_lower_bound(self):
        # Construct vectors that should ideally minimize mutual information
        original_a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        original_b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)

        # Normalize to ensure no scale issues
        a = F.normalize(original_a, p=2, dim=-1)
        b = F.normalize(original_b, p=2, dim=-1)

        # Compute the similarity
        similarity = self.loss_fn.similarity(a, b)
        self.assertAlmostEqual(similarity.item(), 1.0, places=5)

        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

        # # Compute gradients
        # loss.backward()

        # print(original_a.grad)

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_theoretical_upper_bound -v
    def test_theoretical_upper_bound(self):
        a = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        b = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])

        # Normalize to avoid any scale issues
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)

        # Compute the similarity
        similarity = self.loss_fn.similarity(a, b)
        self.assertAlmostEqual(similarity.item(), 0.0, places=5)

        # The expected loss should be close to 0 if mutual information is maximized
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_dimension_mismatch -v
    def test_dimension_mismatch(self):
        a = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        b = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

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

        self.assertAlmostEqual(best_diversity, 1.0, places=5)
        assert best_candidate == c, best_candidate

        best_diversity, best_candidate = pick_most_diverse([b, c], [a, e, d])

        self.assertAlmostEqual(best_diversity, 1.0, places=5)
        assert best_candidate == a, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c], [d, e])

        self.assertAlmostEqual(best_diversity, 0.0, places=5)
        assert best_candidate == d, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c, d], [e])

        self.assertAlmostEqual(best_diversity, 1.0, places=5)
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

        self.assertAlmostEqual(best_diversity, 1.0, places=5)
        assert best_candidate == c, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b], [c, d, e])

        self.assertAlmostEqual(best_diversity, 1.0, places=5)
        assert best_candidate == c, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c], [d, e])

        self.assertAlmostEqual(best_diversity, 0.0, places=5)
        assert best_candidate == d, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c, d], [e])

        self.assertAlmostEqual(best_diversity, 1.0, places=5)
        assert best_candidate == e, best_candidate

    # python -m unittest losses.quaild_log_det_mi_loss_test.TestQuaidLogDetMILoss.test_overfit -v
    def test_overfit(self):
        set_seed(42)

        # embedding_size = 256
        # num_docs = 10
        # batch_size = 2

        # embedding_size = 3
        # num_docs = 4
        # batch_size = 2

        # Create random tensors for a and b
        # original_a = torch.randn(
        #     batch_size, num_docs, embedding_size, requires_grad=False
        # )
        # original_b = torch.randn(
        #     batch_size, num_docs, embedding_size, requires_grad=False
        # )
        # original_a = torch.tensor(
        #     [
        #         [[1.0, 0.01, 0.01], [1.0, 0.01, 0.01]],
        #         [[0.01, 0.01, 1.0], [0.01, 0.01, -1.0]],
        #     ]
        # )
        # original_b = torch.tensor(
        #     [
        #         [[-1.0, 0.01, 0.01], [-1.0, 0.01, 0.01]],
        #         [[0.01, 0.01, -1.0], [0.01, 0.01, 1.0]],
        #     ]
        # )
        # original_a = torch.tensor([[[1.0, 0.1, 0.1]]])
        # original_b = torch.tensor([[[-1.0, 0.1, 0.1]]])

        original_a = torch.tensor([[[1.0, 0.1, 0.1]]])
        original_b = torch.tensor([[[0.1, 1.0, 0.1]]])

        # Normalize a and b
        normalized_a = F.normalize(original_a, dim=-1)
        normalized_b = F.normalize(original_b, dim=-1)

        # Make a and b require gradients by reassigning them as new tensors
        a = normalized_a.clone().detach().requires_grad_(True)
        b = normalized_b.clone().detach().requires_grad_(True)

        optimizer = AdamW([a, b], lr=1e-1)

        # Training loop
        for epoch in range(10):
            optimizer.zero_grad()
            loss = self.loss_fn(a, b)

            # print(loss)

            mse_loss = torch.nn.functional.mse_loss(a, b, reduction="mean")

            loss.backward()

            # Print gradients for inspection
            # print(f"Gradients of a at Epoch {epoch+1}: {a.grad.tolist()}")
            # print(f"Gradients of b at Epoch {epoch+1}: {b.grad.tolist()}")

            optimizer.step()

            # Re-normalize a and b after the update step
            with torch.no_grad():
                a.copy_(F.normalize(a, dim=-1))  # In-place update of 'a'
                b.copy_(F.normalize(b, dim=-1))  # In-place update of 'b'

            print(
                f"Epoch {epoch+1}, Loss: {loss.item()}, MSE: {mse_loss.item()}",
                a.tolist(),
                b.tolist(),
            )

        # assert mse_loss.item() <= 1e-3, mse_loss.item()


if __name__ == "__main__":
    unittest.main()
