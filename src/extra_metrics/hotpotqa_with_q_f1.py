from collections import defaultdict
from extra_metrics.base import ExtraMetricsBase
from extra_metrics.utils import compute_pr_metrics, get_color, pretty_print
import torch


class ExtraMetricHotpotQaWithQF1(ExtraMetricsBase):
    NAME = "hotpot_qa_with_q_f1"

    @staticmethod
    def _count_actually_correct(
        predicted_indices,
        no_paraphrase_idxs,
        paraphrase_lut,
        scores=None,
    ):
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

        # bar, correct_mask = pretty_print(
        #     flipped_predicted_indices, no_paraphrase_idxs, scores
        # )
        # print(bar)
        # print("".join([get_color(value) + " " + "\033[0m" for value in scores]))
        # ALL_SCORES.append(scores)
        # CORRECT_MASKS.append(correct_mask)
        # plot_to_disk()

        return len(set(flipped_predicted_indices).intersection(set(no_paraphrase_idxs)))

    @torch.no_grad()
    def sweeping_pr_curve(self, batch, resolution=100):
        result_score = defaultdict(list)
        result_k = defaultdict(list)
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

            predicted_indices, scores = (
                self.pipeline.subset_selection_strategy.subset_select(
                    question_embedding, document_embeddings
                )
            )
            predicted_indices = predicted_indices.tolist()

            for k in range(1, len(scores) - 1):
                sliced_scores = scores[:k]
                sliced_predicted_indices = predicted_indices[:k]
                next_score = scores[k]
                next_score = int((next_score * resolution * resolution)) // resolution

                num_correct = ExtraMetricHotpotQaWithQF1._count_actually_correct(
                    sliced_predicted_indices,
                    no_paraphrase_idxs,
                    paraphrase_lut,
                    sliced_scores,
                )
                num_shortlisted = len(predicted_indices)
                max_correct = len(no_paraphrase_idxs)

                _, _, f1_score = compute_pr_metrics(
                    num_correct, num_shortlisted, max_correct
                )

                result_score[next_score].append(f1_score)
                result_k[k].append(f1_score)

        return dict(result_score), dict(result_k)

    @torch.no_grad()
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

            predicted_indices, scores = (
                self.pipeline.subset_selection_strategy.subset_select(
                    question_embedding, document_embeddings
                )
            )
            predicted_indices = predicted_indices.tolist()

            num_correct = ExtraMetricHotpotQaWithQF1._count_actually_correct(
                predicted_indices, no_paraphrase_idxs, paraphrase_lut, scores
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
