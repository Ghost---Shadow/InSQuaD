from concurrent.futures import ThreadPoolExecutor
import json
import os
from generative_models.openai_chat_model import OpenAIChatModel
from datasets import load_dataset
from tqdm import tqdm
from functools import lru_cache


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
        for paragraph_index, paragraph in enumerate(row["context"]["content"]):
            for sentence_index, sentence in enumerate(paragraph):
                future = executor.submit(cached_generate_question, sentence)
                futures_dict[(paragraph_index, sentence_index)] = future

        # Organize the results into the structure of paragraphs and sentences
        for paragraph_index, paragraph in enumerate(row["context"]["content"]):
            paragraph_questions = []
            for sentence_index, _ in enumerate(paragraph):
                future = futures_dict[(paragraph_index, sentence_index)]
                paragraph_questions.append(future.result())
            all_questions.append(paragraph_questions)

    row["context"]["questions"] = all_questions

    return row


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


def convert_to_question_for_split(dataset, model, split, debug):
    split_path = f"data/2wikimultihopqa_with_q_gpt35/{split}.jsonl"
    old_split_path = f"data/2wikimultihopqa_with_q_gpt35/{split}_old.jsonl"

    TRAIN_LIMIT = 15000

    processed_ids = set()
    if os.path.exists(split_path):
        with open(split_path) as f:
            for line in f:
                row = json.loads(line)
                processed_ids.add(row["_id"])

    # Load original data for reference
    original_data = {}
    if os.path.exists(old_split_path):
        with open(old_split_path, "r") as f:
            for line in f:
                row = json.loads(line)
                original_data[row["_id"]] = row

    with open(split_path, "a") as f:
        for current_row, row in enumerate(
            tqdm(dataset[split], total=min(TRAIN_LIMIT, len(dataset[split])))
        ):
            if debug and current_row >= 5:
                break

            if row["_id"] in processed_ids:
                continue

            # Limit for trainset for now
            if current_row >= TRAIN_LIMIT:
                break

            # Check and add missing components
            row = original_data.get(row["_id"], row)
            row = add_question_to_row(model, row)
            row = add_paraphrased_question_to_row(model, row)

            f.write(json.dumps(row) + "\n")
            f.flush()


def convert_to_question_dataset(model, debug=False):
    dataset = load_dataset("scholarly-shadows-syndicate/2WikiMultihopQA")

    convert_to_question_for_split(dataset, model, "train", debug)
    convert_to_question_for_split(dataset, model, "dev", debug)


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
