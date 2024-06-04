from datasets import load_dataset
from dataloaders.base import BaseDataset


class HotpotQaDataset(BaseDataset):
    NAME = "hotpot_qa"

    def __init__(self, config):
        super().__init__(config)
        self.dataset = load_dataset("hotpot_qa", "distractor")

    @staticmethod
    def collate_fn(batch, KEY_SENTENCES="sentences"):
        batch_questions = []
        batch_flat_sentences = []
        batch_relevant_sentence_indexes = []
        batch_labels_mask = []
        # batch_flag_for_error = []

        for item in batch:
            # flag_for_error = False
            question = item["question"]
            flat_sentences = []

            index_lut = {}
            for title, sentences in zip(
                item["context"]["title"], item["context"][KEY_SENTENCES]
            ):
                sent_counter = 0

                for sentence in sentences:
                    index_lut[(title, sent_counter)] = len(flat_sentences)
                    flat_sentences.append(sentence)
                    sent_counter += 1

            relevant_sentence_indexes = []
            for title, sent_id in zip(
                item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]
            ):
                key = (title, sent_id)
                if key in index_lut:
                    flat_index = index_lut[key]
                    relevant_sentence_indexes.append(flat_index)
                # else:
                #     flag_for_error = True

            # Sort if necessary
            relevant_sentence_indexes = sorted(relevant_sentence_indexes)

            labels_mask = [False] * len(flat_sentences)
            for index in relevant_sentence_indexes:
                labels_mask[index] = True

            batch_questions.append(question)
            batch_flat_sentences.append(flat_sentences)
            batch_relevant_sentence_indexes.append(relevant_sentence_indexes)
            batch_labels_mask.append(labels_mask)
            # batch_flag_for_error.append(flag_for_error)

        return {
            "question": batch_questions,
            "documents": batch_flat_sentences,
            "relevant_indexes": batch_relevant_sentence_indexes,
            "correct_mask": batch_labels_mask,
            "paraphrase_masks": [],  # No paraphrase for base dataset
            # "flag_for_error": batch_flag_for_error,
        }
