import glob
import json
import shutil

from config import Config
from generative_models.openai_pretrained_model import WrappedOpenAiPretrained
from tqdm import tqdm


def find_broken_files():
    pattern = "artifacts/*/seed_42/xsum/inference_result.jsonl"
    matching_files = glob.glob(pattern)
    bad_files = []
    for file_name in matching_files:
        with open(file_name) as f:
            for line in f:
                row = json.loads(line)
                if "rouge" not in row:
                    bad_files.append(file_name)
                break
    return bad_files


def rougify_row(row, model):
    row["rouge"] = model._compute_rouge(row["target"], row["predicted"])
    return row


if __name__ == "__main__":
    config = Config.from_file(
        "experiments/model_size_ablations/oracle_mpnet_davinci2.yaml"
    )
    model = WrappedOpenAiPretrained(config, config.offline_validation.generative_model)
    bad_files = find_broken_files()

    for file_path in tqdm(bad_files):
        # Make a backup of the file
        backup_path = file_path + ".backup"
        shutil.copy(file_path, backup_path)

        updated_rows = []
        with open(backup_path, "r") as infile:
            for line in infile:
                row = json.loads(line)
                if "rouge" not in row:
                    row = rougify_row(row, model)
                updated_rows.append(row)

        # Write the updated data back to the original file
        with open(file_path, "w") as outfile:
            for row in updated_rows:
                outfile.write(json.dumps(row) + "\n")
