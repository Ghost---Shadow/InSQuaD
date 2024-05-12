class QAWithNewLine:
    NAME = "q_a_with_new_line"

    def __init__(self, config):
        self.config = config

    def generate_prompt(self, tokenizer, row, few_shots):
        model_max_length = tokenizer.model_max_length

        query = row["prompts"]
        label = row["labels"]

        # Initialize prompt with the final question and answer to ensure they fit
        final_q = f"Q: {query}\nA: "
        final_qa = f"{final_q}{label}\n"
        final_qa_tokens = tokenizer(final_qa, add_special_tokens=False)["input_ids"]
        available_tokens = model_max_length - len(final_qa_tokens)

        if available_tokens < 0:
            query_tokens = tokenizer(query)["input_ids"][:available_tokens]
            query = tokenizer.decode(query_tokens)
            final_q = f"Q: {query}\nA: "
            if len(tokenizer(final_q)["input_ids"]) < model_max_length:
                return final_q
            else:
                return ""  # There is nothing we can do

        few_shot_prompt = ""
        for prompt, label in zip(few_shots["prompts"], few_shots["labels"]):
            current_qa = f"Q: {prompt}\nA: {label}\n\n"
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
