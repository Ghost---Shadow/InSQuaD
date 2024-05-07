from extra_metrics.base import ExtraMetricsBase
from extra_metrics.utils import compute_pr_metrics
import torch


def pretty_print(a, b):
    result = ""
    seen = set()  # To track elements seen so far in `a`.

    for index, element in enumerate(a):
        if element in b:
            if element in seen:
                # Element is repeating and is present in `b`.
                result += "\033[93m|\033[0m"
            else:
                # Element is present in `b` and not seen before.
                result += "\033[92m|\033[0m"
                seen.add(element)
        else:
            # Element is not present in `b`.
            result += "\033[91m|\033[0m"

    return result


class ExtraMetricHotpotQaWithQF1(ExtraMetricsBase):
    NAME = "hotpot_qa_with_q_f1"

    @staticmethod
    def _count_actually_correct(predicted_indices, no_paraphrase_idxs, paraphrase_lut):
        """
        If the model and selection picks both the correct answer
        and its paraphrase then only count one as correct
        """
        flipped_predicted_indices = []
        for idx in predicted_indices:
            if idx not in no_paraphrase_idxs:
                flipped_predicted_indices.append(paraphrase_lut[idx])
            else:
                flipped_predicted_indices.append(idx)

        print(pretty_print(flipped_predicted_indices, no_paraphrase_idxs))

        return len(set(flipped_predicted_indices).intersection(set(no_paraphrase_idxs)))

    @torch.no_grad
    def generate_metric(self, batch):
        batch_precision = []
        batch_recall = []
        batch_f1_score = []

        for question, documents, no_paraphrase_idxs, paraphrase_lut in zip(
            batch["question"],
            batch["documents"],
            batch["relevant_indexes"],
            batch["paraphrase_lut"],
        ):
            all_text = [question, *documents]
            all_embeddings = self.pipeline.semantic_search_model.embed(all_text)
            question_embedding = all_embeddings[0]
            document_embeddings = all_embeddings[1:]

            predicted_indices = self.pipeline.subset_selection_strategy.subset_select(
                question_embedding, document_embeddings
            )
            predicted_indices = predicted_indices.tolist()

            num_correct = ExtraMetricHotpotQaWithQF1._count_actually_correct(
                predicted_indices, no_paraphrase_idxs, paraphrase_lut
            )
            num_shortlisted = len(predicted_indices)
            max_correct = len(no_paraphrase_idxs)

            precision, recall, f1_score = compute_pr_metrics(
                num_correct, num_shortlisted, max_correct
            )
            batch_precision.append(precision)
            batch_recall.append(recall)
            batch_f1_score.append(f1_score)

        def mean(arr):
            return sum(arr) / len(arr)

        return {
            "precision": mean(batch_precision),
            "recall": mean(batch_recall),
            "f1_score": mean(batch_f1_score),
        }
