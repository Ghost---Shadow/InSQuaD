from dataloaders.base import BaseDataset
from dataloaders.hotpot_qa_loader import HotpotQaDataset
from datasets import load_dataset
import torch


class HotpotQaWithQDataset(BaseDataset):
    NAME = "hotpot_qa_with_q"

    def __init__(self, config):
        super().__init__(config)
        self.dataset = load_dataset(
            "scholarly-shadows-syndicate/hotpotqa_with_qa_gpt35"
        )

    @staticmethod
    def paraphrase_lut_to_mask(lookup_table, max_length):
        mask_list = []

        all_items = list(sorted(lookup_table.items()))
        items = all_items[: len(all_items) // 2]  # Symmetry

        for key, value in items:
            left_mask = torch.zeros(max_length, dtype=torch.bool)
            right_mask = torch.zeros(max_length, dtype=torch.bool)

            left_mask[key] = True
            right_mask[value] = True

            mask_list.append((left_mask, right_mask))

        return mask_list

    @staticmethod
    def downstream_collate(batch, upstream_batch):
        batch_flat_questions = []
        batch_relevant_question_indexes = []
        batch_correct_mask = []
        batch_paraphrase_lut = []
        batch_paraphrase_masks = []
        batch_upstream_relevant_indexes = upstream_batch["relevant_indexes"]
        batch_upstream_correct_mask = upstream_batch["correct_mask"]
        for item, upstream_relevant_indexes, upstream_correct_mask in zip(
            batch, batch_upstream_relevant_indexes, batch_upstream_correct_mask
        ):
            flat_questions = []
            for paragraph_questions in item["context"]["questions"]:
                for question in paragraph_questions:
                    flat_questions.append(question)

            paraphrased_questions = []
            for paragraph_questions in item["context"]["paraphrased_questions"]:
                for question in paragraph_questions:
                    paraphrased_questions.append(question)

            all_flat_question = [*flat_questions, *paraphrased_questions]
            batch_flat_questions.append(all_flat_question)

            # No paraphrase
            relevant_question_indexes = upstream_relevant_indexes
            batch_relevant_question_indexes.append(relevant_question_indexes)

            # Merge selection vector
            downstream_correct_mask = upstream_correct_mask
            correct_mask = [*upstream_correct_mask, *downstream_correct_mask]
            batch_correct_mask.append(torch.tensor(correct_mask))

            # Paraphrase look up table
            offset = len(flat_questions)
            paraphrase_lut = {}
            for idx in range(offset):
                paraphrase_idx = offset + idx
                paraphrase_lut[paraphrase_idx] = idx
                paraphrase_lut[idx] = paraphrase_idx
            batch_paraphrase_lut.append(paraphrase_lut)
            paraphrase_masks = HotpotQaWithQDataset.paraphrase_lut_to_mask(
                paraphrase_lut, len(all_flat_question)
            )
            batch_paraphrase_masks.append(paraphrase_masks)

        return {
            "upstream_documents": upstream_batch["documents"],
            "documents": batch_flat_questions,
            "relevant_indexes": batch_relevant_question_indexes,
            "correct_mask": batch_correct_mask,
            "paraphrase_lut": batch_paraphrase_lut,
            "paraphrase_masks": batch_paraphrase_masks,
        }

    @staticmethod
    def collate_fn(batch):
        upstream_batch = HotpotQaDataset.collate_fn(batch)
        downstream_batch = HotpotQaWithQDataset.downstream_collate(
            batch, upstream_batch
        )

        return {
            **upstream_batch,
            **downstream_batch,
        }
