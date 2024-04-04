from abc import abstractmethod
from transformers import AutoTokenizer, AutoModel


class WrappedBaseModel:
    def __init__(self, config):
        self.config = config

        checkpoint = config.architecture.semantic_search_model.checkpoint
        self.device = config.architecture.semantic_search_model.device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)

    def get_all_trainable_parameters(self):
        return self.model.parameters()

    @abstractmethod
    def embed(self, sentences): ...
