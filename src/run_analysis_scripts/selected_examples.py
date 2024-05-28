import json
import re


def count_correct_answers(file_path):
    correct_count = 0
    yes_count = 0
    full_agreement_and_correct = 0
    full_agreement_and_incorrect = 0
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

    return (
        correct_count,
        yes_count,
        full_agreement_and_correct,
        full_agreement_and_incorrect,
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
    left_corrects, left_yes_count, left_full_correct, left_full_incorrect = (
        count_correct_answers(left_file_path)
    )
    right_corrects, right_yes_count, right_full_correct, right_full_incorrect = (
        count_correct_answers(right_file_path)
    )

    print(
        f"Full aggreement on correct left: {left_full_correct}/{left_corrects} ({left_full_correct/left_corrects:.4f})"
    )
    print(
        f"Full aggreement on correct right: {right_full_correct}/{right_corrects} ({right_full_correct/right_corrects:.4f})"
    )
    left_incorrects = 256 - left_corrects
    right_incorrects = 256 - right_corrects
    print(
        f"Full aggreement on incorrect left {left_full_incorrect}/{left_incorrects} ({left_full_incorrect/left_incorrects:.4f})"
    )
    print(
        f"Full aggreement on incorrect right {right_full_incorrect}/{right_incorrects} ({right_full_incorrect/right_incorrects:.4f})"
    )

    print(f"Number of yes answers in left file: {left_yes_count}")
    print(f"Number of yes answers in right file: {right_yes_count}")


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


CHERRY_PICKS = [
    {
        "left_file_path": "artifacts/quaild_nt_gc_mpnet_gemma_360ba/seed_42/mrpc/inference_result.jsonl",
        "right_file_path": "artifacts/quaild_gain_gc_mpnet_gemma_lambda_025_3c10e/seed_42/mrpc/inference_result.jsonl",
        "left_model_name": "QuailD-GC (NT)",
        "right_model_name": "QuailD-GC ($\\lambda = 0.25$)",
        "skip_not_all_agree": True,
        "label": "nt_vs_t_all_agree_fail",
        "caption": "An example where there was a full label agreement, but the generative model still decided on the wrong answer. (5-shot followed by actual prompt and generated answer, correct answer is 'yes')",
        "skips": 0,
    }
]


if __name__ == "__main__":
    for cherry_pick in CHERRY_PICKS:
        # File paths
        left_file_path = cherry_pick["left_file_path"]
        right_file_path = cherry_pick["right_file_path"]
        label = cherry_pick["label"]
        caption = cherry_pick["caption"]
        skip_not_all_agree = cherry_pick["skip_not_all_agree"]
        left_model_name = cherry_pick["left_model_name"]
        right_model_name = cherry_pick["right_model_name"]
        skips = cherry_pick["skips"]

        output_file_path = f"artifacts/tables/{label}.tex"

        # Read both files
        left_data = read_jsonl(left_file_path)
        right_data = read_jsonl(right_file_path)

        # Find the first instance where left is wrong and right is correct
        for left, right in zip(left_data, right_data):
            if not left["correct"] and right["correct"]:
                if skips > 0:
                    skips -= 1
                    continue

                if skip_not_all_agree:
                    few_shot_labels = re.findall(r"A: (.*?)\n", left["prompts"])
                    all_agree = all(
                        label == left["labels"] for label in few_shot_labels
                    )
                    if not all_agree:
                        continue

                # with open("debug.txt", "w") as out_file:
                #     # json.dump({"left": left, "right": right}, out_file, indent=4)
                #     out_file.write(left["prompts"] + pick_actual(left))
                #     out_file.write("\n" + ("-" * 80) + "\n")
                #     out_file.write(right["prompts"] + pick_actual(right))

                with open(output_file_path, "w") as f:
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
