import json
import os
import math
from openai import OpenAI
import tiktoken


class WrappedOpenAiPretrained:
    NAME = "openai_pretrained"

    def __init__(self, config, generative_model_config):
        self.config = config
        self.generative_model_config = generative_model_config
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = generative_model_config.checkpoint
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def evaluate_with_options(self, prompt, correct_option_index, options):
        # Construct the prompt with enumerated options
        full_prompt = (
            f"{prompt}\nOptions:\n"
            + "\n".join([f"{idx+1}: {option}" for idx, option in enumerate(options)])
            + "\nA: "
        )
        response = self.client.completions.create(
            model=self.model_name,
            prompt=full_prompt,
            temperature=0.5,
            max_tokens=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=5,  # Get log probabilities for the top 5 tokens
        )
        response = response.model_dump()

        # Extract the log probabilities for the completion's token
        token_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

        # Convert log probabilities to probabilities and match them with the correct options
        option_probabilities = {}
        for i, option in enumerate(options):
            token_key = str(i + 1)  # Matching index to token keys which are 1-based
            if token_key in token_logprobs:
                option_probabilities[option] = math.exp(token_logprobs[token_key])

        # Normalize probabilities to sum to 1
        total_prob = sum(option_probabilities.values())
        option_probabilities = {
            option: prob / total_prob for option, prob in option_probabilities.items()
        }

        # Determine if the most likely option is the correct one
        most_likely_option = max(option_probabilities, key=option_probabilities.get)
        most_likely_option_index = options.index(most_likely_option)
        is_correct = most_likely_option_index == correct_option_index

        return {
            "option_probabilities": option_probabilities,
            "correct": is_correct,
        }
