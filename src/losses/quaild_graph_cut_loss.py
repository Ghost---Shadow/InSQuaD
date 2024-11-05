from losses.base_loss import BaseLoss
import torch


class QuaildGraphCutLoss(BaseLoss):
    NAME = "graph_cut"

    def __init__(self, config):
        super(QuaildGraphCutLoss, self).__init__(config)
        self.lambd = config.training.loss.lambd

    def similarity(self, d, q):
        # d.shape = [batch_size, num_docs_d, embedding_size]
        # q.shape = [batch_size, num_docs_q, embedding_size]

        dq_similarities = self.compute_similarity_matrix(d, q)
        return self.similarity_matrix_to_information(None, dq_similarities, None)

    def similarity_matrix_to_information(
        self, qq_similarities, dq_similarities, dd_similarities
    ):
        # similarities.shape = [batch_size, num_docs_d, num_docs_q]
        similarities = dq_similarities

        # Normalize and rescale
        batch_size, num_docs_a, num_docs_b = similarities.shape
        similarities = similarities.reshape([batch_size, num_docs_a * num_docs_b])
        aggregated_similarity = 2 * self.lambd * similarities.mean(dim=-1)

        return aggregated_similarity

    def forward(self, a, b):
        loss = -self.similarity(a, b)

        # Adjust the lower bound for each item in the batch
        theoretical_lower_bound = 2 * self.lambd * -1
        loss = loss - theoretical_lower_bound

        # Should never be negative at this point, so mean along batch
        return loss.mean()
