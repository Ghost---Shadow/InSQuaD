from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

# from generative_models.wrapped_t5 import WrappedT5
from generative_models.openai_chat_model import OpenAIChatModel
import json
import os
from tqdm import tqdm
from functools import lru_cache


def add_paraphrased_question_to_row(model, row):
    # Already computed
    if "paraphrased_questions" in row["context"]:
        return row

    @lru_cache(maxsize=1024)
    def cached_generate_paraphrase(sentence):
        return model.generate_paraphrase(sentence)

    all_questions = []

    # Create a single ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        # Store futures for each sentence in a dictionary to maintain order
        futures_dict = {}
        for paragraph_index, paragraph in enumerate(row["context"]["questions"]):
            for sentence_index, sentence in enumerate(paragraph):
                future = executor.submit(cached_generate_paraphrase, sentence)
                futures_dict[(paragraph_index, sentence_index)] = future

        # Organize the results into the structure of paragraphs and questions
        for paragraph_index, paragraph in enumerate(row["context"]["questions"]):
            paragraph_questions = []
            for sentence_index, _ in enumerate(paragraph):
                future = futures_dict[(paragraph_index, sentence_index)]
                paragraph_questions.append(future.result())
            all_questions.append(paragraph_questions)

    row["context"]["paraphrased_questions"] = all_questions

    return row


def add_question_to_row(model, row):
    # Already computed
    if "questions" in row["context"]:
        return row

    @lru_cache(maxsize=1024)
    def cached_generate_question(sentence):
        return model.generate_question(sentence)

    all_questions = []

    # Create a single ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        # Store futures for each sentence in a dictionary to maintain order
        futures_dict = {}
        for paragraph_index, paragraph in enumerate(row["context"]["sentences"]):
            for sentence_index, sentence in enumerate(paragraph):
                future = executor.submit(cached_generate_question, sentence)
                futures_dict[(paragraph_index, sentence_index)] = future

        # Organize the results into the structure of paragraphs and sentences
        for paragraph_index, paragraph in enumerate(row["context"]["sentences"]):
            paragraph_questions = []
            for sentence_index, _ in enumerate(paragraph):
                future = futures_dict[(paragraph_index, sentence_index)]
                paragraph_questions.append(future.result())
            all_questions.append(paragraph_questions)

    row["context"]["questions"] = all_questions

    return row


def convert_to_question_for_split(dataset, model, split, debug):
    old_split_path = f"./data/hotpotqa_with_qa_gpt35/{split}_old.jsonl"
    new_split_path = f"./data/hotpotqa_with_qa_gpt35/{split}.jsonl"

    # Load existing data from the new file
    processed_ids = set()
    if os.path.exists(new_split_path):
        with open(new_split_path, "r") as f:
            for line in f:
                row = json.loads(line)
                processed_ids.add(row["id"])  # Assuming each row has a unique 'id'

    # Load original data for reference
    original_data = {}
    if os.path.exists(old_split_path):
        with open(old_split_path, "r") as f:
            for line in f:
                row = json.loads(line)
                original_data[row["id"]] = row

    TRAIN_LIMIT = 15000

    # Process and append new rows
    with open(new_split_path, "a") as new_file:
        for current_row, row in enumerate(
            tqdm(dataset[split], total=min(TRAIN_LIMIT, len(dataset[split])))
        ):
            # Skip already processed rows
            if row["id"] in processed_ids:
                continue

            if debug and current_row >= 100:
                break

            # Limit for trainset for now
            if current_row >= TRAIN_LIMIT:
                break

            # Check and add missing components
            row = original_data.get(row["id"], row)
            row = add_question_to_row(model, row)
            row = add_paraphrased_question_to_row(model, row)

            new_file.write(json.dumps(row) + "\n")
            new_file.flush()


def convert_to_question_dataset(model, debug=False):
    dataset = load_dataset("hotpot_qa", "distractor")

    convert_to_question_for_split(dataset, model, "train", debug)
    convert_to_question_for_split(dataset, model, "validation", debug)


if __name__ == "__main__":
    config = {
        "architecture": {
            "question_generator_model": {
                "name": "gpt-3.5-turbo",
                # "name": "t5",
                # "size": "base",
                # "device": "cuda:0",
            }
        }
    }
    # model = T5ModelForQuestionGeneration(config)
    model = OpenAIChatModel(config)
    convert_to_question_dataset(model, debug=False)
