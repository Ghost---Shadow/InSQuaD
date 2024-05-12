import os


def write_inference_done(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "final_result.json" in filenames:
            final_result_path = os.path.join(dirpath, "final_result.json")
            if os.path.isfile(final_result_path):
                inference_done_path = os.path.join(dirpath, "inference.done")
                open(inference_done_path, "w").close()

                few_shot_done_path = os.path.join(dirpath, "few_shot.done")
                open(few_shot_done_path, "w").close()


if __name__ == "__main__":
    write_inference_done("./artifacts")
