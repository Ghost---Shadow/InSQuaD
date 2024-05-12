from collections import defaultdict
from typing import List
from dataloaders.base import BaseDataset


def get_options_if_possible(wrapped_dataset: BaseDataset):
    if hasattr(wrapped_dataset, "LABELS") and wrapped_dataset.LABELS is not None:
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


def flatten_batch_of_batches(batch_of_batches):
    result = defaultdict(list)
    for batch in batch_of_batches:
        for key in batch:
            result[key] = [*result[key], *batch[key]]

    return dict(result)
