import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F


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

    def evaluate(self, prompt, target_answer):
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        # Tokenize the target answer and add special tokens (e.g., EOS) as needed by the model
        target_answer_ids = self.tokenizer(
            target_answer, add_special_tokens=False
        ).input_ids
        target_answer_ids = torch.tensor(
            [target_answer_ids], device=self.device
        ).squeeze()

        # Conduct the forward pass and calculate the loss automatically
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=target_answer_ids.shape[0],
            output_scores=True,
            return_dict_in_generate=True,
        )

        # [sequence_length, vocab_size]
        scores = torch.stack(outputs.scores).squeeze()

        # Softmax to convert logits to probabilities
        probabilities = F.softmax(scores, dim=-1)

        # Gather the log probabilities of the actual target tokens
        target_probs = torch.gather(
            probabilities, index=target_answer_ids.unsqueeze(-1), dim=-1
        ).squeeze(-1)
        target_log_probs = torch.log(target_probs)

        # Sum of log probabilities for the target sequence
        target_log_likelihood = target_log_probs.sum()

        # Exponential of the negative sum of log probabilities gives the sequence probability
        target_sequence_probability = torch.exp(target_log_likelihood).item()

        # Similarly for predicted, if using max for most probable tokens
        _, output_tokens = torch.max(scores, dim=-1)
        predicted_probs = torch.gather(
            probabilities, index=output_tokens.unsqueeze(-1), dim=-1
        ).squeeze(-1)
        predicted_log_probs = torch.log(predicted_probs)
        predicted_sequence_probability = torch.exp(predicted_log_probs.sum()).item()

        # Sanity check
        redecoded_target = self.tokenizer.decode(
            target_answer_ids, skip_special_tokens=True
        )
        predicted = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

        return {
            "target_sequence_probability": target_sequence_probability,
            "predicted_sequence_probability": predicted_sequence_probability,
            "target": redecoded_target,
            "predicted": predicted,
        }
