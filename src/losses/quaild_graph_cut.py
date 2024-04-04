import torch
import torch.nn as nn


class QuaildGraphCut(nn.Module):
    def __init__(self, config):
        super(QuaildGraphCut, self).__init__()
        self.lambd = config.training.loss.lambd

    def forward(self, a, b):
        # Ensuring a and b are 2D and have compatible dimensions for batch matrix multiplication
        # Assuming a and b are of shape [batch_size, features], where each row is a vector.

        # Compute the dot product between all pairs.
        # For dot products, we can use matrix multiplication.
        # First, let's ensure a is [batch_size, features] and b is [features, batch_size] for matrix multiplication with a.
        # Transpose b to make its shape compatible for matrix multiplication with a.
        b_t = b.transpose(0, 1)

        # Matrix multiplication
        # This will give us a [batch_size, batch_size] tensor where each element is a dot product of vectors from a and b.
        similarity = torch.matmul(a, b_t)

        # Since the question implies summing all combinations, we sum all elements in the resulting matrix.
        loss = 2 * self.lambd * similarity.sum()
        return loss
