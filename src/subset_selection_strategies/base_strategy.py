class BaseSubsetSelectionStrategy:
    @staticmethod
    def apply_indexes(retrieved_documents, indexes):
        # Initialize lists to store the subsets of documents
        subset_prompts = []
        subset_labels = []
        subset_distances = []

        # Iterate over each document and corresponding index
        for doc, idx in zip(retrieved_documents, indexes):
            # Select subsets based on the indexes for this document
            selected_prompts = [doc["prompts"][i] for i in idx]
            selected_labels = [doc["labels"][i] for i in idx]
            selected_distances = [doc["distances"][i] for i in idx]

            # Append the selected subsets to their respective lists
            subset_prompts.append(selected_prompts)
            subset_labels.append(selected_labels)
            subset_distances.append(selected_distances)

        # Create a new list of documents with the selected subsets
        selected_documents = [
            {
                "prompts": prompts,
                "labels": labels,
                "distances": distances,
            }
            for prompts, labels, distances in zip(
                subset_prompts, subset_labels, subset_distances
            )
        ]

        return selected_documents
