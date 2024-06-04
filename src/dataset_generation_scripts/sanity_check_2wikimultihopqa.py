import json
from datasets import load_dataset
from tqdm import tqdm


def check_ids_in_jsonl(file_path, split):
    dataset = load_dataset("[anonymized]", split=split)
    seen_rows = 0
    expected_length = min(15000, len(dataset))
    with open(file_path, "r") as f:
        for line, actual_row in tqdm(zip(f, dataset), total=expected_length):
            row = json.loads(line)
            assert actual_row["_id"] == row["_id"]
            seen_rows += 1

    assert seen_rows == expected_length, seen_rows


if __name__ == "__main__":
    file_path = "data/2wikimultihopqa_with_q_gpt35/dev.jsonl"
    split = "dev"
    check_ids_in_jsonl(file_path, split)

    file_path = "data/2wikimultihopqa_with_q_gpt35/train.jsonl"
    split = "train"
    check_ids_in_jsonl(file_path, split)
