import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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


def get_color(value):
    if value < 0.1:
        return "\033[48;2;128;0;0m"  # Dark Red
    elif value < 0.2:
        return "\033[48;2;255;0;0m"  # Red
    elif value < 0.3:
        return "\033[48;2;255;165;0m"  # Orange
    elif value < 0.4:
        return "\033[48;2;255;255;0m"  # Yellow
    elif value < 0.5:
        return "\033[48;2;154;205;50m"  # Yellow-Green
    elif value < 0.6:
        return "\033[48;2;0;255;0m"  # Green
    elif value < 0.7:
        return "\033[48;2;0;191;255m"  # Deep Sky Blue
    elif value < 0.8:
        return "\033[48;2;0;0;255m"  # Blue
    elif value < 0.9:
        return "\033[48;2;75;0;130m"  # Indigo
    elif value < 0.95:
        return "\033[48;2;238;130;238m"  # Violet
    elif value < 0.99:
        return "\033[48;2;255;182;193m"  # Light Pink
    else:
        return "\033[48;2;255;255;255m"  # White


def pretty_print(a, b, scores):
    result = ""
    seen = set()  # To track elements seen so far in `a`.

    last_correct_score = -1
    correct_mask = []

    for index, (element, score) in enumerate(zip(a, scores)):
        if element in b:
            if element in seen:
                # Element is repeating and is present in `b`.
                result += "\033[93m|\033[0m"
                correct_mask.append(0)
            else:
                # Element is present in `b` and not seen before.
                result += "\033[92m|\033[0m"
                last_correct_score = score
                seen.add(element)
                correct_mask.append(1)
        else:
            # Element is not present in `b`.
            result += "\033[91m|\033[0m"
            correct_mask.append(1)

    result += f"({last_correct_score:.4f})"

    return result, correct_mask


def plot_to_disk(all_scores, correct_masks):
    data = {"Epoch": [], "Score": [], "Document": [], "Style": []}

    # STYLES = {
    #     0: "+",
    #     1: "s",
    #     2: "x",
    # }

    for i, (scores, masks) in enumerate(zip(all_scores, correct_masks)):
        for Document, (score, mask) in enumerate(zip(scores, masks)):
            data["Epoch"].append(i)
            data["Score"].append(score)
            data["Document"].append(Document)
            data["Style"].append(mask)

    df = pd.DataFrame(data)

    sns.set_theme("paper")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Document",
        y="Score",
        hue="Epoch",
        # style="Style",
        # markers=STYLES,
    )
    plt.title("Scores over Training Epochs")
    plt.xlabel("Document")
    plt.ylabel("Score")

    # Save the plot to disk
    plot_path = "training_scores_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path
