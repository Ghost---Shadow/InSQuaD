class NoOp:
    NAME = "noop"

    def __init__(self, *args, **kwargs): ...

    def batch_evaluate(self, *args, **kwargs):
        raise NotImplementedError("Should not be invoked")

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("Should not be invoked")
