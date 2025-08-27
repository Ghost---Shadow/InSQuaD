class PromptFormattingBaseStrategy:
    def __init__(self, config):
        self.config = config
        self.final_q_format = None
        self.final_qa_format = None
        self.shot_format = None

    def generate_prompt(self, tokenizer, row, few_shots):
        model_max_length = tokenizer.model_max_length

        query = row["prompts"]
        label = row["labels"]

        # Initialize prompt with the final question and answer to ensure they fit
        final_q = self.final_q_format.format(query=query)
        final_qa = self.final_qa_format.format(final_q=final_q, label=label)
        final_qa_tokens = tokenizer(final_qa, add_special_tokens=False)["input_ids"]
        available_tokens = model_max_length - len(final_qa_tokens)

        if available_tokens < 0:
            query_tokens = tokenizer(query)["input_ids"][:available_tokens]
            query = tokenizer.decode(query_tokens)
            final_q = self.final_q_format.format(query=query)
            if len(tokenizer(final_q)["input_ids"]) < model_max_length:
                return final_q
            else:
                return ""  # There is nothing we can do

        few_shot_prompt = ""
        for prompt, label in zip(few_shots["prompts"], few_shots["labels"]):
            current_qa = self.shot_format.format(prompt=prompt, label=label)
            current_qa_tokens = tokenizer(current_qa, add_special_tokens=False)[
                "input_ids"
            ]
            if len(current_qa_tokens) <= available_tokens:
                few_shot_prompt += current_qa
                available_tokens -= len(current_qa_tokens)
            else:
                # Stop if adding another few-shot example exceeds max length
                break

        # Append the final question and answer at the end
        few_shot_prompt += final_q

        return few_shot_prompt
