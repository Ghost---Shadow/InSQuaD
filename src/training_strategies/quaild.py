# from config import RootConfig
import torch


class QuaildStrategy:
    def __init__(self, config, pipeline):
        self.pipeline = pipeline
        self.config = config

    def before_each_epoch(self): ...

    def train_step(self, batch):
        total_loss = 0
        for idx in range(len(batch["question"])):
            # Gradient accumulation
            total_loss += self.train_step_inner(batch, idx)

        return total_loss

    def train_step_inner(self, batch, idx):
        question = batch["question"][idx]
        documents = batch["documents"][idx]
        correct_mask = batch["correct_mask"][idx]
        paraphrase_masks = batch["paraphrase_masks"][idx]

        all_text = [question, *documents]

        all_embeddings = self.pipeline.semantic_search_model.embed(all_text)

        # question_embedding.shape = [1, embedding_dim]
        question_embedding = all_embeddings[0].unsqueeze(0)
        # document_embeddings.shape = [num_docs, embedding_dim]
        document_embeddings = all_embeddings[1:]

        correct_embeddings = document_embeddings[correct_mask]
        incorrect_embeddings = document_embeddings[~correct_mask]

        gq_plus = self.pipeline.loss_function(correct_embeddings, question_embedding)
        gq_minus = self.pipeline.loss_function(incorrect_embeddings, question_embedding)

        all_gd_plus = []
        all_gd_minus = []
        for paraphrase_mask in paraphrase_masks:
            left_mask, right_mask = paraphrase_mask
            actual_embeddings = document_embeddings[left_mask]
            paraphrase_embeddings = document_embeddings[right_mask]
            not_paraphrase_embeddings = document_embeddings[~right_mask]
            inner_gd_plus = self.pipeline.loss_function(
                actual_embeddings, paraphrase_embeddings
            )
            inner_gd_minus = self.pipeline.loss_function(
                actual_embeddings, not_paraphrase_embeddings
            )

            if torch.isfinite(torch.log(inner_gd_plus)) and torch.isfinite(
                torch.log(inner_gd_minus)
            ):
                all_gd_plus.append(inner_gd_plus)
                all_gd_minus.append(inner_gd_minus)

        # gd_plus.shape = [num_docs]
        gd_plus = torch.stack(all_gd_plus)
        gd_minus = torch.stack(all_gd_minus)

        loss_q = (torch.log(gq_minus) - torch.log(gq_plus)).mean()
        loss_d = (torch.log(gd_minus) - torch.log(gd_plus)).mean()

        lambdA = self.config.training.loss.lambd

        loss = loss_q + lambdA * loss_d

        # Lower bound it to 0
        loss = torch.exp(loss)

        return loss
