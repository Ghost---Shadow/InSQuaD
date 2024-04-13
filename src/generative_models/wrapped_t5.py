import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class WrappedT5:
    def __init__(self, config, generative_model_config):
        self.config = config
        self.generative_model_config = generative_model_config
        checkpoint = generative_model_config.checkpoint
        device = generative_model_config.device
        self.device = torch.device(device)

        # Load the model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    def batch_evaluate(self, batch_prompt, batch_true_answer):
        # TODO: Single pass
        results = []
        for prompt, true_answer in zip(batch_prompt, batch_true_answer):
            results.append(self.evaluate(prompt, true_answer))

        return torch.tensor(results).to(self.device)

    def evaluate(self, prompt, true_answer):
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        # Tokenize the true answer and add special tokens (e.g., EOS) as needed by the model
        true_answer_ids = self.tokenizer(true_answer, add_special_tokens=True).input_ids
        true_answer_ids = torch.tensor([true_answer_ids], device=self.device)

        # Conduct the forward pass and calculate the loss automatically
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                labels=true_answer_ids,
            )
            loss = outputs.loss

        actual = self.tokenizer.decode(torch.argmax(outputs.logits))

        sequence_probability = torch.exp(-loss).item()

        return {
            "sequence_probability": sequence_probability,
            "actual": actual,
        }
