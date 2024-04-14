import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class WrappedAutoModel:
    def __init__(self, config, generative_model_config):
        self.config = config
        self.generative_model_config = generative_model_config
        checkpoint = generative_model_config.checkpoint
        device = generative_model_config.device
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

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
            outputs = self.model.forward(
                input_ids=input_ids,
                labels=true_answer_ids,
            )
            loss = outputs.loss

        output_tokens = torch.argmax(outputs.logits.squeeze(), dim=-1)
        actual = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

        sequence_probability = torch.exp(-loss).item()

        return {
            "sequence_probability": sequence_probability,
            "actual": actual,
        }
