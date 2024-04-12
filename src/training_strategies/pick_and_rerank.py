import torch


class PickAndRerankStrategy:
    def __init__(self, config, pipeline):
        self.pipeline = pipeline
        self.config = config

    def before_each_epoch(self):
        wrapped_train_dataset = self.pipeline.wrapped_train_dataset
        semantic_search_model = self.pipeline.semantic_search_model
        self.pipeline.dense_index.repopulate_index(
            wrapped_train_dataset, semantic_search_model
        )

    def train_step(self, batch):
        questions = batch["prompts"]
        labels = batch["labels"]

        # Search FAISS for similar questions
        question_embeddings = self.pipeline.semantic_search_model.embed(questions)
        retrieved_documents = self.pipeline.dense_index.retrieve(question_embeddings)

        # Compute the quality diversity terms
        quality_vector, diversity_matrix = (
            self.pipeline.semantic_search_model.compute_quality_diversity(
                question_embeddings, retrieved_documents
            )
        )

        # Rerank selected questions and pick a subset
        indexes = self.pipeline.subset_selection_strategy.get_indexes_from_qd(
            quality_vector, diversity_matrix
        )
        fewshot_subset = self.pipeline.subset_selection_strategy.apply_indexes(
            retrieved_documents, indexes
        )

        # Turn it into a fewshot prompt
        tokenizer = self.pipeline.generative_model.tokenizer
        fewshot_prompt = self.pipeline.prompt_formatting_strategy.generate_prompt(
            tokenizer, batch, fewshot_subset
        )

        # Compute sequence_probability with labels (no grad)
        with torch.no_grad():
            batch_sequence_probability = self.pipeline.generative_model.batch_evaluate(
                fewshot_prompt, labels
            )

        # Step 5: Compute loss for semantic search model
        mean_quality = quality_vector.mean(dim=-1)
        loss = self.pipeline.loss_function(mean_quality, batch_sequence_probability)

        return loss
