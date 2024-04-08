from concurrent.futures import ThreadPoolExecutor
import json
from models.openai_chat_model import OpenAIChatModel


def add_paraphrases_to_row(model, row):
    # Already computed
    if "paraphrased_questions" in row["context"]:
        return row

    def generate_paraphrase(sentence):
        return model.generate_paraphrase(sentence)

    all_questions = []

    # Create a single ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        # Store futures for each sentence in a dictionary to maintain order
        futures_dict = {}
        for paragraph_index, paragraph in enumerate(row["context"]["sentences"]):
            for sentence_index, sentence in enumerate(paragraph):
                future = executor.submit(generate_paraphrase, sentence)
                futures_dict[(paragraph_index, sentence_index)] = future

        # Organize the results into the structure of paragraphs and sentences
        for paragraph_index, paragraph in enumerate(row["context"]["sentences"]):
            paragraph_questions = []
            for sentence_index, _ in enumerate(paragraph):
                future = futures_dict[(paragraph_index, sentence_index)]
                paragraph_questions.append(future.result())
            all_questions.append(paragraph_questions)

    row["context"]["questions_paraphrased"] = all_questions

    return row


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
    file_name = "src/one_off_experiments/paraphrase_sample.json"
    with open(file_name) as f:
        row = json.load(f)
    row = add_paraphrases_to_row(model, row)

    with open(file_name, "w") as f:
        json.dump(row, f, indent=2)
