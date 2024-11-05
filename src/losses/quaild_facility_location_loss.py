from losses.base_loss import BaseLoss
import torch


class QuaildFacilityLocationLoss(BaseLoss):
    NAME = "facility_location"

    def __init__(self, config):
        """
        Initialize the FacilityLocationLoss module.

        Parameters:
        - lambd (float): Regularization parameter.
        """
        super(QuaildFacilityLocationLoss, self).__init__(config)
        self.lambd = config.training.loss.lambd
        self.epsilon = 0.0  # Adjust if needed

    def similarity(self, a, b):
        # d.shape = [batch_size, num_docs_d, embedding_size]
        # q.shape = [batch_size, num_docs_q, embedding_size]

        dq_similarities = self.compute_similarity_matrix(a, b)
        return self.similarity_matrix_to_information(None, dq_similarities, None)

    def similarity_matrix_to_information(
        self, qq_similarities, dq_similarities, dd_similarities
    ):
        similarities = dq_similarities

        # Compute max similarity for each i in a with all j in b
        # [batch_size, num_docs, num_docs] to [batch_size, num_docs]
        max_similarities_a_to_b, _ = torch.max(similarities, dim=1)
        max_similarities_b_to_a, _ = torch.max(similarities, dim=2)

        # Sum these max similarities and apply the regularization term
        mean_a_to_b = max_similarities_a_to_b.mean(dim=-1)
        mean_b_to_a = max_similarities_b_to_a.mean(dim=-1)
        mean_max_similarities = mean_a_to_b + self.lambd * mean_b_to_a

        # Upper bound is now 1
        mean_max_similarities = mean_max_similarities / (1 + self.lambd)

        return mean_max_similarities

    def forward(self, a, b):
        """
        Compute the Facility Location loss between two sets of embeddings.

        Parameters:
        - a (Tensor): The embeddings tensor of shape (batch_size, num_docs, embedding_size).
        - b (Tensor): The embeddings tensor of shape (batch_size, num_docs, embedding_size),
                      representing a different set or the same set as 'a'.

        Returns:
        - loss (Tensor): The computed Facility Location loss.
        """
        # similarity returns value in range [0,1]
        loss = 1 - self.similarity(a, b)

        # Should never be negative at this point, so mean along batch
        return loss.mean()
