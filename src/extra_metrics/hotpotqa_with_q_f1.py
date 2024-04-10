from extra_metrics.base import ExtraMetricsBase
from extra_metrics.utils import compute_pr_metrics
import torch


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

        return len(set(flipped_predicted_indices).intersection(set(no_paraphrase_idxs)))

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

        def tensorify_mean(arr):
            return torch.tensor(arr, dtype=torch.float32).mean().item()

        return {
            "precision": tensorify_mean(batch_precision),
            "recall": tensorify_mean(batch_recall),
            "f1_score": tensorify_mean(batch_f1_score),
        }
