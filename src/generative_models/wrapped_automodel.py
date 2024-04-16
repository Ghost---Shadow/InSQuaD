import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


class WrappedAutoModel:
    NAME = "automodel"

    def __init__(self, config, generative_model_config):
        self.config = config
        self.generative_model_config = generative_model_config
        checkpoint = generative_model_config.checkpoint
        device = generative_model_config.device
        self.device = torch.device(device)
        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    def evaluate_with_options(self, prompt, correct_option_index, options):
        # Tokenize input
        all_text = [prompt, *options]
        all_encoded = self.tokenizer.batch_encode_plus(all_text)
        prompt_input_ids = torch.tensor(
            all_encoded["input_ids"][0], device=self.device
        ).unsqueeze(0)
        prompt_attention_mask = torch.tensor(
            all_encoded["attention_mask"][0], device=self.device
        ).unsqueeze(0)
        options_input_ids = all_encoded["input_ids"][1:]

        longest_option_len = max(
            len(option_input_ids) for option_input_ids in options_input_ids
        )

        outputs = self.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=longest_option_len,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # [sequence_length, batch_size, vocab_size] to [sequence_length, vocab_size]
        scores = torch.stack(outputs.scores).squeeze(1)

        # Softmax to convert logits to probabilities
        probabilities = F.softmax(scores, dim=-1)

        # Gather the log probabilities of the actual target tokens
        options_sequence_probability = []
        for option_input_ids in options_input_ids:
            option_input_ids = torch.tensor(
                option_input_ids, device=probabilities.device
            )
            option_probs = torch.gather(
                probabilities, index=option_input_ids.unsqueeze(-1), dim=-1
            ).squeeze(-1)
            option_log_probs = torch.log(option_probs)

            # Sum of log probabilities for the option sequence
            option_log_likelihood = option_log_probs.sum()

            # Exponential of the negative sum of log probabilities gives the sequence probability
            option_sequence_probability = torch.exp(option_log_likelihood)
            options_sequence_probability.append(option_sequence_probability)
        options_sequence_probability = torch.stack(options_sequence_probability)
        options_sequence_probability = options_sequence_probability / torch.norm(
            options_sequence_probability
        )

        predicted_option = torch.argmax(options_sequence_probability)
        option_probabilities = {}
        for option, probability in zip(options, options_sequence_probability):
            option_probabilities[option] = probability.item()

        return {
            "option_probabilities": option_probabilities,
            "correct": (predicted_option == correct_option_index).item(),
        }

    def evaluate(self, prompt, target_answer):
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        # Tokenize the target answer and add special tokens (e.g., EOS) as needed by the model
        target_answer_ids = self.tokenizer(
            target_answer, add_special_tokens=False
        ).input_ids
        target_answer_ids = torch.tensor(target_answer_ids, device=self.device)

        # Conduct the forward pass and calculate the loss automatically
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
            max_new_tokens=target_answer_ids.shape[0],
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # [sequence_length, batch_size, vocab_size] to [sequence_length, vocab_size]
        scores = torch.stack(outputs.scores).squeeze(1)

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
