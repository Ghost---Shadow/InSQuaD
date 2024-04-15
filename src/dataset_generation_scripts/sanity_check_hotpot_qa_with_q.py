import json
from datasets import load_dataset
from tqdm import tqdm


def check_ids_in_jsonl(file_path, split):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    seen_rows = 0
    expected_length = min(15000, len(dataset))
    with open(file_path, "r") as f:
        for line, actual_row in tqdm(zip(f, dataset), total=expected_length):
            row = json.loads(line)
            assert actual_row["id"] == row["id"]
            seen_rows += 1

    assert seen_rows == expected_length, seen_rows


if __name__ == "__main__":
    file_path = "data/hotpotqa_with_qa_gpt35/validation.jsonl"
    split = "validation"
    check_ids_in_jsonl(file_path, split)

    file_path = "data/hotpotqa_with_qa_gpt35/train.jsonl"
    split = "train"
    check_ids_in_jsonl(file_path, split)
