from abc import abstractmethod


class ExtraMetricsBase:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @abstractmethod
    def generate_metric(self, batch):
        ...
