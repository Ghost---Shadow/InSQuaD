import torch
from losses.base_loss import BaseLoss


class QuaidLogDetMILoss(BaseLoss):
    NAME = "log_det_mi"

    def __init__(self, config):
        super(QuaidLogDetMILoss, self).__init__()
        self.lambd = config.training.loss.lambd
        self.epsilon = 1e-4
        self.positive_inf = 1e6

    def finitify(self, x):
        """
        Helper function to smoothly limit values within a tensor using tanh, maintaining differentiability.
        """
        # Check for and handle infinity
        x = torch.where(x.abs() == float("inf"), torch.sign(x) * self.positive_inf, x)
        return x

    def logdetMI(self, S_A, S_AQ, S_Q):
        # Register a hook to trace gradients
        # def bind_print_grad(name):
        #     def print_grad(grad):
        #         print(name, grad)

        #     return print_grad

        # Print input tensors
        # print("S_A", S_A)
        # print("S_AQ", S_AQ)
        # print("S_Q", S_Q)

        # Bounds [-inf, 0]
        log_det_SA = torch.logdet(S_A)
        # log_det_SA.register_hook(bind_print_grad("log_det_SA"))
        log_det_SA = self.finitify(log_det_SA)
        # print("log_det_SA", log_det_SA)

        S_Q_inv = torch.pinverse(S_Q)
        # S_Q_inv.register_hook(bind_print_grad("S_Q_inv"))
        # print("S_Q_inv", S_Q_inv)

        second_term = S_A - self.lambd**2 * S_AQ @ S_Q_inv @ S_AQ.transpose(1, 2)
        # second_term.register_hook(bind_print_grad("second_term"))
        # print("second_term", second_term)

        log_det_second = torch.logdet(second_term)
        # log_det_second.register_hook(bind_print_grad("log_det_second"))
        log_det_second = self.finitify(log_det_second)
        # print("log_det_second", log_det_second)

        log_mutual_information = log_det_SA - log_det_second
        # log_mutual_information.register_hook(bind_print_grad("log_mutual_information"))
        log_mutual_information = self.finitify(log_mutual_information)
        # log_mutual_information.register_hook(
        # bind_print_grad("log_mutual_information_finitify")
        # )
        # print("log_det_SA", "log_det_second", log_det_SA, log_det_second)
        # print("log_mutual_information", log_mutual_information)

        mutual_information = torch.exp(log_mutual_information)
        # mutual_information.register_hook(bind_print_grad("mutual_information"))
        mutual_information = self.finitify(mutual_information)
        # mutual_information.register_hook(bind_print_grad("mutual_information"))
        # print("mutual_information", mutual_information)

        # Return scaled mutual information
        return mutual_information / self.positive_inf

    def compute_similarity(self, a, b):
        """
        Compute a bounded similarity between tensors 'a' and 'b' using matrix multiplication.
        Assumes inputs are normalized.
        """
        # Use matrix multiplication to compute similarity
        similarity = torch.matmul(a, b.transpose(1, 2))

        # Bounded between -1 and 1
        return similarity

    def ensure_3d(self, tensor):
        """
        Ensure the input tensor has three dimensions [batch_size, num_vectors, vector_dimension].
        If the tensor is 2D, interpret it as [1, num_vectors, vector_dimension].
        """
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def similarity(self, a, b):
        # Ensure inputs are 3D
        a = self.ensure_3d(a)
        b = self.ensure_3d(b)

        # Compute the similarity matrices
        S_A = self.compute_similarity(a, a)
        S_AQ = self.compute_similarity(a, b)
        S_Q = self.compute_similarity(b, b)

        # Compute the mutual information as a loss
        mutual_information = self.logdetMI(S_A, S_AQ, S_Q)
        # Bounds [0, 1]
        return mutual_information

    def forward(self, a, b):
        a = self.ensure_3d(a)
        b = self.ensure_3d(b)

        similarity = self.similarity(a, b)

        # The loss is 1 - similarity to minimize the loss while maximizing MI
        # Bounds [0, 1]
        return (1 - similarity).mean(dim=0)
