class WrappedOpenAiPretrained:
    NAME = "openai_pretrained"

    def __init__(self, config, generative_model_config):
        self.config = config
        self.generative_model_config = generative_model_config
        # TODO
