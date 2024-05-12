from prompt_formatting_strategies.base import PromptFormattingBaseStrategy


class QAWithNewLine(PromptFormattingBaseStrategy):
    NAME = "q_a_with_new_line"

    def __init__(self, config):
        super().__init__(config)

        self.final_q_format = "Q: {query}\nA: "
        self.final_qa_format = "{final_q}{label}\n"
        self.shot_format = "Q: {prompt}\nA: {label}\n\n"
