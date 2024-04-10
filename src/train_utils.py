import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def average_dicts(dict_list):
    """
    Averages a list of dictionaries with numerical values.
    Assumes all dictionaries in the list have the same structure.

    Parameters:
    - dict_list: List[Dict[str, float]]

    Returns:
    - Dict[str, float]: Dictionary with the same keys as the input dictionaries and their average values.
    """
    if not dict_list:
        return {}

    # Initialize a dictionary to store summed values
    sum_dict = {key: 0 for key in dict_list[0].keys()}

    # Sum values from all dictionaries
    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] += value

    # Average the summed values
    avg_dict = {key: value / len(dict_list) for key, value in sum_dict.items()}

    return avg_dict
