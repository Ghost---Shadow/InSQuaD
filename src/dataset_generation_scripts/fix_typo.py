import json


def rename_key_in_jsonl(input_file, output_file, old_key, new_key):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            data["context"][new_key] = data["context"].pop(old_key)
            assert old_key not in data["context"]
            json.dump(data, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    # Paths to the original and new files
    input_files = [
        "data/hotpotqa_with_qa_gpt35/potato/train_broken.jsonl.no",
        "data/hotpotqa_with_qa_gpt35/potato/validation_broken.jsonl.no",
    ]
    output_files = [
        "data/hotpotqa_with_qa_gpt35/train.jsonl",
        "data/hotpotqa_with_qa_gpt35/validation.jsonl",
    ]

    # Process each file
    for in_file, out_file in zip(input_files, output_files):
        rename_key_in_jsonl(
            in_file, out_file, "questions_paraphrased", "paraphrased_questions"
        )

    print("Processing complete.")
