class QAWithNewLine:
    def __init__(self, config):
        self.config = config

    def generate_prompt(self, tokenizer, batch, batch_few_shot_batch):
        final_prompts = []
        model_max_length = tokenizer.model_max_length

        for query, label, few_shot in zip(
            batch["prompts"], batch["labels"], batch_few_shot_batch
        ):
            # Initialize prompt with the final question and answer to ensure they fit
            final_q = f"Q: {query}\nA: "
            final_qa = f"{final_q}{label}\n"
            final_qa_tokens = tokenizer(final_qa, add_special_tokens=False)["input_ids"]
            available_tokens = model_max_length - len(final_qa_tokens)

            few_shot_prompt = ""
            for prompt, label in zip(few_shot["prompts"], few_shot["labels"]):
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
            final_prompts.append(few_shot_prompt)

        return final_prompts
