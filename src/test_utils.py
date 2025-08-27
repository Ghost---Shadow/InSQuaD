import torch


def skip_if_no_gpu(test_func):
    """Decorator to skip test if GPU is not available"""

    def wrapper(self):
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        return test_func(self)

    return wrapper
