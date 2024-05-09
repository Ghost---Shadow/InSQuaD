import unittest
from losses.utils import bind_pick_most_diverse
import torch
from torch.optim import AdamW
from losses.quaild_graph_cut_loss import QuaildGraphCutLoss
from config import Config
from train_utils import set_seed
import torch.nn.functional as F


# python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut -v
class TestQuaildGraphCut(unittest.TestCase):
    def setUp(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        self.loss_fn = QuaildGraphCutLoss(config)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_theoretical_lower_bound -v
    def test_theoretical_lower_bound(self):
        a = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        b = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
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
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 1.0)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_should_not_cancel_out -v
    def test_should_not_cancel_out(self):
        a = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ]
        )
        b = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            ]
        )
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 0.5)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_dimension_mismatch -v
    def test_dimension_mismatch(self):
        a = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )
        b = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        loss = self.loss_fn(a, b)
        self.assertAlmostEqual(loss.item(), 0.0)

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_submodularity -v
    def test_submodularity(self):
        # q = [0.7071, 0.7071, 0.0000] # query
        a = [1.0000, 0.0000, 0.0000]  # 0 # partial match
        b = [0.7071, 0.7071, 0.0000]  # 1 # perflect quality, first pick
        c = [0.0000, 0.0000, 1.0000]  # 2 # completely orthogonal
        d = [0.7071, 0.7071, 0.0000]  # 3 # perfect quality, wrong diversity
        e = [-0.7071, -0.7071, 0.0000]  # 4 # anti-parallel quality

        pick_most_diverse = bind_pick_most_diverse(self)

        best_diversity, best_candidate = pick_most_diverse([b], [a, c, d, e])

        assert best_diversity == 1.0, best_diversity
        assert best_candidate == e, best_candidate

        best_diversity, best_candidate = pick_most_diverse([b, e], [a, c, d])

        assert best_diversity == 0.5, best_diversity
        assert best_candidate == a, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, e], [c, d])

        assert best_diversity == 0.5, best_diversity
        assert best_candidate == c, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c, e], [d])

        assert best_diversity == 0.4116116762161255, best_diversity
        assert best_candidate == d, best_candidate

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_submodularity_with_arbitary_order -v
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

        assert best_diversity == 1.0, best_diversity
        assert best_candidate == e, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b], [c, d, e])

        assert best_diversity == 0.9267767667770386, best_diversity
        assert best_candidate == e, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c], [d, e])

        assert best_diversity == 0.7845178246498108, best_diversity
        assert best_candidate == e, best_candidate

        best_diversity, best_candidate = pick_most_diverse([a, b, c, d], [e])

        assert best_diversity == 0.8383883833885193, best_diversity
        assert best_candidate == e, best_candidate

    # python -m unittest losses.quaild_graph_cut_loss_test.TestQuaildGraphCut.test_overfit -v
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
        original_a = torch.tensor(
            [
                [[1.0, 0.01, 0.01], [1.0, 0.01, 0.01]],
                [[0.01, 0.01, 1.0], [0.01, 0.01, -1.0]],
            ]
        )
        original_b = torch.tensor(
            [
                [[-1.0, 0.01, 0.01], [-1.0, 0.01, 0.01]],
                [[0.01, 0.01, -1.0], [0.01, 0.01, 1.0]],
            ]
        )
        # original_a = torch.tensor([[[1.0, 0.1, 0.1]]])
        # original_b = torch.tensor([[[-1.0, 0.1, 0.1]]])

        # Normalize a and b
        normalized_a = F.normalize(original_a, dim=-1)
        normalized_b = F.normalize(original_b, dim=-1)

        # Make a and b require gradients by reassigning them as new tensors
        a = normalized_a.clone().detach().requires_grad_(True)
        b = normalized_b.clone().detach().requires_grad_(True)

        optimizer = AdamW([a, b], lr=1)

        # Training loop
        for epoch in range(10):
            optimizer.zero_grad()
            loss = self.loss_fn(a, b)

            mse_loss = torch.nn.functional.mse_loss(a, b, reduction="mean")

            loss.backward()
            optimizer.step()

            # Re-normalize a and b after the update step
            with torch.no_grad():
                a.copy_(F.normalize(a, dim=-1))  # In-place update of 'a'
                b.copy_(F.normalize(b, dim=-1))  # In-place update of 'b'

            # print(
            #     f"Epoch {epoch+1}, Loss: {loss.item()}, MSE: {mse_loss.item()}",
            #     # a.tolist(),
            #     # b.tolist(),
            # )

        assert mse_loss.item() <= 1e-3, mse_loss.item()


if __name__ == "__main__":
    unittest.main()
