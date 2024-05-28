from collections import Counter
import json
import math
import re


def calculate_entropy(data):
    # Count the frequency of each label in the data
    label_counts = {}
    for label in data:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # Calculate the total number of items
    total_items = len(data)

    # Calculate the entropy
    entropy = 0
    for label in label_counts:
        probability = label_counts[label] / total_items
        entropy -= probability * math.log2(probability)

    # Calculate maximum entropy
    num_classes = len(label_counts)
    max_entropy = (
        math.log2(num_classes) if num_classes > 1 else 1
    )  # Handle the case with 1 class

    # Normalize the entropy
    normalized_entropy = (
        entropy / max_entropy if max_entropy else 0
    )  # Avoid division by zero

    return normalized_entropy


def count_correct_answers(file_path):
    correct_count = 0
    yes_count = 0
    full_agreement_and_correct = 0
    full_agreement_and_incorrect = 0
    total_entropy = 0
    label_majority_win = 0
    total = 0
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["correct"] is True:
                correct_count += 1
            if data["labels"] == "yes":
                yes_count += 1

            few_shot_labels = re.findall(r"A: (.*?)\n", data["prompts"])
            all_agree = all(label == data["labels"] for label in few_shot_labels)
            if data["correct"] and all_agree:
                full_agreement_and_correct += 1

            if all_agree and not data["correct"]:
                full_agreement_and_incorrect += 1

            most_common = Counter(few_shot_labels).most_common(1)[0][0]
            if most_common == data["labels"]:
                label_majority_win += 1

            total_entropy += calculate_entropy(few_shot_labels)
            total += 1

    entropy = total_entropy / total

    return (
        correct_count,
        yes_count,
        full_agreement_and_correct,
        full_agreement_and_incorrect,
        entropy,
        label_majority_win,
        total,
    )


def read_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def pick_actual(row):
    option_probabilities = row["option_probabilities"]
    # "option_probabilities": {"no": 0.6144433617591858, "yes": 0.7889609336853027}
    return max(option_probabilities, key=option_probabilities.get)


def print_statistics(left_file_path, right_file_path):
    # Counting correct answers
    (
        left_corrects,
        left_yes_count,
        left_full_correct,
        left_full_incorrect,
        left_entropy,
        left_label_majority_win,
        left_total,
    ) = count_correct_answers(left_file_path)
    (
        right_corrects,
        right_yes_count,
        right_full_correct,
        right_full_incorrect,
        right_entropy,
        right_label_majority_win,
        right_total,
    ) = count_correct_answers(right_file_path)

    assert left_total == right_total

    print(
        f"Full aggreement on correct left: {left_full_correct}/{left_corrects} ({left_full_correct/left_corrects:.4f})"
    )
    print(
        f"Full aggreement on correct right: {right_full_correct}/{right_corrects} ({right_full_correct/right_corrects:.4f})"
    )
    left_incorrects = left_total - left_corrects
    right_incorrects = right_total - right_corrects
    print(
        f"Full aggreement on incorrect left {left_full_incorrect}/{left_incorrects} ({left_full_incorrect/left_incorrects:.4f})"
    )
    print(
        f"Full aggreement on incorrect right {right_full_incorrect}/{right_incorrects} ({right_full_incorrect/right_incorrects:.4f})"
    )

    print(
        f"Label majority win left {left_label_majority_win}/{left_total} ({left_label_majority_win/left_total:.4f})"
    )
    print(
        f"Label majority win right {right_label_majority_win}/{right_total} ({right_label_majority_win/right_total:.4f})"
    )

    print(f"Number of yes answers in left file: {left_yes_count}")
    print(f"Number of yes answers in right file: {right_yes_count}")

    print(f"entropy: {left_entropy}:{right_entropy}")


def write_latex_table(
    left, right, file, left_model_name, right_model_name, label, caption
):
    # Replace '&' and '%' in LaTeX with escaped versions to avoid errors
    left_prompts = left["prompts"].replace("&", "\\&").replace("%", "\\%").split("Q: ")
    right_prompts = (
        right["prompts"].replace("&", "\\&").replace("%", "\\%").split("Q: ")
    )

    file.write("\\begin{table}[H]\n")
    file.write("\\centering\n")
    file.write("\\small\n")
    file.write(f"\caption{{{caption}}}\n")
    file.write(f"\\label{{table:{label}}}\n")
    file.write("\\begin{tabular}{|p{0.47\\textwidth}|p{0.47\\textwidth}|}\n")
    file.write("\\hline\n")
    file.write(f"\\textbf{{{left_model_name}}} & \\textbf{{{right_model_name}}} \\\\\n")
    file.write("\\hline\n")

    # Handling multiple questions, assuming that left and right are matched in count
    for idx, (left_prompt, right_prompt) in enumerate(zip(left_prompts, right_prompts)):
        if left_prompt and right_prompt:

            if idx == len(left_prompts) - 1:
                left_answer = pick_actual(left)
                right_answer = pick_actual(right)
                file.write(
                    "\\textbf{Q:} "
                    + left_prompt.strip()
                    + left_answer
                    + " & \\textbf{Q:} "
                    + right_prompt.strip()
                    + right_answer
                    + " \\\\\n"
                )
            else:
                file.write(
                    "\\textbf{Q:} "
                    + left_prompt.strip()
                    + " & \\textbf{Q:} "
                    + right_prompt.strip()
                    + " \\\\\n"
                )

            file.write("\\hline\n")

    file.write("\\end{tabular}\n")
    file.write("\\end{table}\n")


def criterion_only_right_correct(left, right):
    return not left["correct"] and right["correct"]


def criterion_only_left_correct(left, right):
    return left["correct"] and not right["correct"]


def criterion_both_correct(left, right):
    return left["correct"] and right["correct"]


def criterion_all_agree(left, right):
    for side in [left, right]:
        few_shot_labels = re.findall(r"A: (.*?)\n", side["prompts"])
        all_agree = all(label == side["labels"] for label in few_shot_labels)
        if not all_agree:
            return False
    return True


def criterion_right_correct_and_all_agree(left, right):
    return criterion_all_agree(left, right) and criterion_only_right_correct(
        left, right
    )


CHERRY_PICKS = [
    {
        "left_file_path": "artifacts/quaild_nt_gc_mpnet_gemma_360ba/seed_42/mrpc/inference_result.jsonl",
        "right_file_path": "artifacts/quaild_gain_gc_mpnet_gemma_lambda_025_3c10e/seed_42/mrpc/inference_result.jsonl",
        "left_model_name": "QuailD-GC (NT)",
        "right_model_name": "QuailD-GC ($\\lambda = 0.25$)",
        "label": "nt_vs_t_all_agree_fail",
        "caption": "An example where there was a full label agreement, but the generative model still decided on the wrong answer. (5-shot followed by actual prompt and generated answer, correct answer is 'yes')",
        "skips": 0,
        "criterion": criterion_right_correct_and_all_agree,
    },
    {
        "left_file_path": "artifacts/quaild_gain_ld_mpnet_gemma_lambda_0_60b87/seed_42/rte/inference_result.jsonl",
        "right_file_path": "artifacts/quaild_gain_ld_mpnet_gemma_lambda_1_bfd38/seed_42/rte/inference_result.jsonl",
        "left_model_name": "QuailD-LD ($\\lambda = 0.0$)",
        "right_model_name": "QuailD-LD ($\\lambda = 1.0$)",
        "label": "effect_of_lambda_rte",
        "caption": "An example in RTE dataset showing the effect of $\\lambda$ on few shot selection where both models got the answer correct.",
        "skips": 0,
        "criterion": criterion_both_correct,
    },
]


if __name__ == "__main__":
    for cherry_pick in CHERRY_PICKS:
        # File paths
        left_file_path = cherry_pick["left_file_path"]
        right_file_path = cherry_pick["right_file_path"]
        label = cherry_pick["label"]
        caption = cherry_pick["caption"]
        left_model_name = cherry_pick["left_model_name"]
        right_model_name = cherry_pick["right_model_name"]
        skips = cherry_pick["skips"]
        criterion = cherry_pick["criterion"]

        output_file_path = f"artifacts/tables/{label}.tex"

        print("-" * 80)
        print(label)
        print_statistics(left_file_path, right_file_path)

        # Read both files
        left_data = read_jsonl(left_file_path)
        right_data = read_jsonl(right_file_path)

        # Find the first instance where left is wrong and right is correct
        for left, right in zip(left_data, right_data):
            if criterion(left, right):
                if skips > 0:
                    skips -= 1
                    continue

                with open(f"./debug_{label}.txt", "w", encoding="utf-8") as out_file:
                    # json.dump({"left": left, "right": right}, out_file, indent=4)
                    out_file.write(left["prompts"] + pick_actual(left))
                    out_file.write("\n" + ("-" * 80) + "\n")
                    out_file.write(right["prompts"] + pick_actual(right))

                with open(output_file_path, "w", encoding="utf-8") as f:
                    write_latex_table(
                        left,
                        right,
                        f,
                        left_model_name,
                        right_model_name,
                        label,
                        caption,
                    )

                break
