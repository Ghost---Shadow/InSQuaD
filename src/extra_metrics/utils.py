def compute_pr_metrics(num_correct, num_shortlisted, max_correct):
    """
    Compute Precision, Recall, and F1 Score.

    Parameters:
    - num_correct: Number of correct documents the model selected.
    - num_shortlisted: Number of documents selected.
    - max_correct: Number of relevant documents.

    Returns:
    A tuple containing the Precision, Recall, and F1 Score.
    """
    # Ensure that the denominator is not 0 to avoid division by zero errors
    precision = num_correct / num_shortlisted if num_shortlisted > 0 else 0
    recall = num_correct / max_correct if max_correct > 0 else 0

    # Calculate F1 Score, ensuring no division by zero
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return precision, recall, f1_score
