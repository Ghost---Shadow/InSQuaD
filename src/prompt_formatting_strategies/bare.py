from prompt_formatting_strategies.base import PromptFormattingBaseStrategy


class BareStrategy(PromptFormattingBaseStrategy):
    NAME = "bare"

    def __init__(self, config):
        super().__init__(config)
        self.final_q_format = "{query}"
        self.final_qa_format = "{final_q}{label}"
        self.shot_format = "{prompt}{label}"
