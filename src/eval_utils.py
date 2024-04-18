from typing import List
from dataloaders.base import BaseDataset


def get_options_if_possible(wrapped_dataset: BaseDataset):
    if hasattr(wrapped_dataset, "LABELS"):
        options = list(wrapped_dataset.LABELS.values())
    else:
        options = None

    return options


def evaluate_with_options_if_possible(
    generative_model, options: List[str], prompt: str, true_answer: str
):
    if options is None:
        result = generative_model.evaluate(prompt, true_answer)
    else:
        correct_option_index = options.index(true_answer)
        result = generative_model.evaluate_with_options(
            prompt, correct_option_index, options
        )

    return result
