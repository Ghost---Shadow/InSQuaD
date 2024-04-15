from transformers import AutoTokenizer

CHECKPOINTS_NEED_TESTING = [
    "stabilityai/stablelm-2-1_6b",
    "google/gemma-2b",
]


def test_if_one_token(DatasetClass):
    labels = DatasetClass.LABELS.values()

    for checkpoint in CHECKPOINTS_NEED_TESTING:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        for label in labels:
            tokens = tokenizer(label, add_special_tokens=False).input_ids
            assert type(tokens) == list, tokens
            assert len(tokens) == 1, (label, tokens)
