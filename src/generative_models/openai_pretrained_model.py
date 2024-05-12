import json
import os
import math
from openai import OpenAI
import tiktoken


class WrappedTokenizer:
    def __init__(self, model_name):
        self.openai_tokenizer = tiktoken.encoding_for_model(model_name)

        attributes_to_copy = [
            attr
            for attr in dir(self.openai_tokenizer)
            if callable(getattr(self.openai_tokenizer, attr))
            and not attr.startswith("__")
        ]

        for attribute in attributes_to_copy:
            setattr(self, attribute, getattr(self.openai_tokenizer, attribute))

        # https://community.openai.com/t/does-the-text-davinci-003-model-support-4000-or-4096-tokens/89507
        self.model_max_length = 4000

    def __call__(self, *args, **kwargs):
        del kwargs["add_special_tokens"]
        return {
            "input_ids": self.encode(*args, **kwargs),
        }


class WrappedOpenAiPretrained:
    NAME = "openai_pretrained"

    def __init__(self, config, generative_model_config):
        self.config = config
        self.generative_model_config = generative_model_config
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = generative_model_config.checkpoint
        self.tokenizer = WrappedTokenizer(self.model_name)
        self.model = None  # Dont remove

    def evaluate_with_options(self, prompt, correct_option_index, options):
        # Construct the prompt with enumerated options
        full_prompt = (
            f"{prompt}\nOptions:\n"
            + "\n".join([f"{idx+1}: {option}" for idx, option in enumerate(options)])
            + "\nA: "
        )

        # Calculate the maximum tokens required for the indices and potential responses
        max_tokens = 0
        for idx in range(len(options)):
            # Ensure to account for both the number as a token and the possible length of the response
            encoded_index = self.tokenizer.encode(f"{idx+1}")
            max_tokens = max(max_tokens, len(encoded_index))

        response = self.client.completions.create(
            model=self.model_name,
            prompt=full_prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=5,  # Get log probabilities for the top 5 tokens
        )
        response = response.model_dump()

        # Extract the log probabilities for the completion's token
        token_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

        # Initialize probabilities for each option to zero
        option_probabilities = {option: 0 for option in options}

        # Update probabilities with those obtained from logprobs
        for i, option in enumerate(options):
            token_key = str(i + 1)  # Matching index to token keys which are 1-based
            if token_key in token_logprobs:
                option_probabilities[option] = math.exp(token_logprobs[token_key])

        # Normalize probabilities to sum to 1 if any probabilities are non-zero
        total_prob = sum(option_probabilities.values())
        if total_prob > 0:
            option_probabilities = {
                option: prob / total_prob
                for option, prob in option_probabilities.items()
            }
        else:
            # Just let it be zero
            ...

        # Determine if the most likely option is the correct one
        most_likely_option = max(option_probabilities, key=option_probabilities.get)
        most_likely_option_index = options.index(most_likely_option)
        is_correct = most_likely_option_index == correct_option_index

        # If all are wrong, then max does not matter
        is_correct = is_correct and total_prob > 0

        return {
            "option_probabilities": option_probabilities,
            "correct": is_correct,
        }

    def evaluate(self, prompt, target_answer):
        encoded_target_answer = self.tokenizer.encode(target_answer)
        max_tokens = len(encoded_target_answer)

        # Send the encoded prompt to the model and get a prediction
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=5,
        )
        response = response.model_dump()

        predicted = response["choices"][0]["text"]
        encoded_predicted = self.tokenizer.encode(predicted)

        # Decode sequences to human-readable format if needed
        redecoded_target = self.tokenizer.decode(encoded_target_answer)
        predicted = self.tokenizer.decode(encoded_predicted)

        log_predicted_sequence_probability = sum(
            response["choices"][0]["logprobs"]["token_logprobs"]
        )
        predicted_sequence_probability = math.exp(log_predicted_sequence_probability)

        return {
            "target_sequence_probability": -1,  # Not possible with API
            "predicted_sequence_probability": predicted_sequence_probability,
            "target": redecoded_target,
            "predicted": predicted,
        }
