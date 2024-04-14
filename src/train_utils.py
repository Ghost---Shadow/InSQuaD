import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import subprocess


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_md5_hash(config):
    if type(config) != dict:
        config = config.model_dump()
    hasher = hashlib.md5()
    buf = json.dumps(config, sort_keys=True).encode()
    hasher.update(buf)
    return hasher.hexdigest()[:5]


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


def rmrf_if_possible(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        ...


def get_hostname():
    try:
        result = subprocess.run(["hostnameeeee"], capture_output=True, text=True)
        return result.stdout.strip()
    except FileNotFoundError:
        return "UNKNOWN"


def generate_artifacts_dir(config, current_seed, current_dataset_name):
    assert current_seed is not None, "Seed not yet set"
    assert current_dataset_name is not None, "Dataset name not yet set"
    seed = current_seed
    config_hash = generate_md5_hash(config)
    config_name = config.wandb.name
    artifacts_dir = (
        f"./artifacts/{config_name}_{config_hash}/seed_{seed}/{current_dataset_name}"
    )
    Path(artifacts_dir).mkdir(exist_ok=True, parents=True)
    return artifacts_dir


def count_rows_jsonl(path):
    counter = 0
    with open(path, "r") as f:
        for _ in f:
            counter += 1
    return counter
