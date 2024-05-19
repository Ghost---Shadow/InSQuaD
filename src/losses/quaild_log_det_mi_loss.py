import torch
from losses.base_loss import BaseLoss
from torch.cuda.amp import autocast


class QuaidLogDetMILoss(BaseLoss):
    NAME = "log_det_mi"

    def __init__(self, config):
        super(QuaidLogDetMILoss, self).__init__(config)
        self.lambd = config.training.loss.lambd
        self.device = config.architecture.semantic_search_model.device
        self.epsilon = torch.tensor(1e-4, device=self.device)
        self.positive_inf = torch.tensor(1e4, device=self.device)
        self.grad_norm_max = torch.tensor(10.0, device=self.device)
        # torch.log(self.positive_inf) crashes
        self.log_positive_inf = torch.tensor(9.21, device=self.device)

    def clamp_gradients_hook(self, grad):
        """
        A method to clamp gradients based on the class's maximum gradient norm.
        """
        norm = torch.norm(grad, 2)
        if norm > self.grad_norm_max:
            return grad * self.grad_norm_max / norm
        return grad

    def attach_gradient_hooks(self, local_vars):
        """
        Helper function to attach gradient hooks to tensors that require gradients.
        """
        for var_name, tensor in local_vars.items():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(self.clamp_gradients_hook)

    def safe_logdet(self, x, name="matrix"):
        """
        Computes the logarithm of the determinant by computing the determinant first,
        then taking the log, and clamping after each step to avoid extreme results.
        """
        # self.print_and_bind(x, "x")

        # Regularize the matrix by adding a small value to its diagonal
        regularized_x = x + self.epsilon * torch.eye(x.size(-1), device=x.device)
        # self.print_and_bind(regularized_x, "regularized_x")

        # Compute the determinant
        det = torch.det(regularized_x)
        # self.print_and_bind(det, "det")

        # Clamp the determinant to avoid negative or zero values that cause issues with log
        clamped_det = torch.clamp(det, min=self.epsilon, max=self.positive_inf)
        # self.print_and_bind(clamped_det, "clamped_det")

        # Compute the logarithm of the clamped determinant
        log_det = torch.log(clamped_det)
        # self.print_and_bind(log_det, "log_det")

        # Clamp the logarithm of the determinant to avoid extreme negative values
        clamped_log_det = torch.clamp(
            log_det, min=-self.log_positive_inf, max=self.log_positive_inf
        )
        # self.print_and_bind(clamped_log_det, "clamped_log_det")

        # Use the helper function to attach hooks
        self.attach_gradient_hooks(locals())

        return clamped_log_det

    def safe_pinverse(self, x, name="matrix"):
        """
        Safe version of torch.pinverse with logging and a gradient hook for debugging.
        Ensures the operation uses float32 precision for stability.
        """

        regularized_x = x + self.epsilon * torch.eye(x.size(-1), device=x.device)
        # self.print_and_bind(regularized_x, "regularized_x")
        clamped_regularized_x = torch.clamp(
            regularized_x, min=-self.positive_inf, max=self.positive_inf
        )
        # self.print_and_bind(clamped_regularized_x, "clamped_regularized_x")

        pinv = torch.pinverse(clamped_regularized_x)
        # self.print_and_bind(pinv, "pinv")

        clamped_pinv = torch.clamp(pinv, min=-self.positive_inf, max=self.positive_inf)
        # self.print_and_bind(clamped_pinv, "clamped_pinv")

        self.attach_gradient_hooks(locals())

        return clamped_pinv

    def safe_exp(self, x, name="value"):
        """
        Safe version of torch.exp that clamps the input to prevent overflow and registers a gradient hook.
        """
        clamped_x = torch.clamp(
            x, min=-self.log_positive_inf, max=self.log_positive_inf
        )
        # self.print_and_bind(clamped_x, "clamped_x")
        exp_x = torch.exp(clamped_x)
        # self.print_and_bind(exp_x, "exp_x")
        clamped_exp_x = torch.clamp(
            exp_x, min=-self.positive_inf, max=self.positive_inf
        )
        # self.print_and_bind(clamped_exp_x, "clamped_exp_x")

        self.attach_gradient_hooks(locals())

        return clamped_exp_x

    def print_and_bind(self, tensor, name):
        def bind_print_grad(name):
            def print_grad(grad):
                print(f"grad_{name}", grad)

            return print_grad

        print(name, tensor)
        tensor.register_hook(bind_print_grad(name))

    def logdetMI(self, S_A, S_AQ, S_Q):
        # print("S_A", S_A)
        # print("S_AQ", S_AQ)
        # print("S_Q", S_Q)

        # Compute log determinant in a safe manner
        log_det_SA = self.safe_logdet(S_A, "S_A")
        # self.print_and_bind(log_det_SA, "log_det_SA")

        S_Q_inv = self.safe_pinverse(S_Q, "S_Q")
        # self.print_and_bind(S_Q_inv, "S_Q_inv")

        rhs_a = S_AQ @ S_Q_inv
        # self.print_and_bind(rhs_a, "rhs_a")
        rhs_b = rhs_a @ S_AQ.transpose(1, 2)
        # self.print_and_bind(rhs_b, "rhs_b")
        second_term = S_A - self.lambd**2 * rhs_b
        # self.print_and_bind(second_term, "second_term")

        log_det_second = self.safe_logdet(second_term, "second_term")
        # self.print_and_bind(log_det_second, "log_det_second")

        log_mutual_information = log_det_SA - log_det_second
        # self.print_and_bind(log_mutual_information, "log_mutual_information")

        mutual_information = self.safe_exp(log_mutual_information, "mutual_information")
        # self.print_and_bind(mutual_information, "mutual_information")

        # Return scaled mutual information
        scaled_mutual_information = mutual_information / self.positive_inf

        self.attach_gradient_hooks(locals())

        return scaled_mutual_information

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

        with autocast(dtype=torch.float32):
            a = a.to(torch.float32)
            b = b.to(torch.float32)

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

        with autocast(dtype=torch.float32):
            a = a.to(torch.float32)
            b = b.to(torch.float32)
            similarity = self.similarity(a, b)

        # The loss is 1 - similarity to minimize the loss while maximizing MI
        # Bounds [0, 1]
        return (1 - similarity).mean(dim=0)
