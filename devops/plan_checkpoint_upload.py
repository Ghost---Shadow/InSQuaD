# import datetime
import os
import glob


def find_latest_checkpoint(experiment_dir):
    """
    Finds the latest checkpoint for a given experiment and seed.
    """
    latest_checkpoints = []
    for seed_dir in glob.glob(f"{experiment_dir}seed_*/"):
        checkpoint_pattern = f"{seed_dir}epoch_*.pth"
        list_of_files = glob.glob(checkpoint_pattern)
        # for file_path in list_of_files:
        #     modification_time = os.path.getmtime(file_path)
        #     readable_time = datetime.datetime.fromtimestamp(modification_time)
        #     print(file_path, readable_time)
        if list_of_files:
            latest_checkpoint = max(list_of_files, key=os.path.getmtime)
            latest_checkpoints.append(latest_checkpoint)
    return latest_checkpoints


if __name__ == "__main__":
    # Base path for checkpoints
    base_path = "./checkpoints/"
    bucket_name = "quaild-icl-bucket"
    shell_script_file = "./devops/upload_checkpoints.sh"

    with open(shell_script_file, "w") as script:
        script.write("#!/bin/bash\n\n")
        for experiment_dir in glob.glob(f"{base_path}*/"):
            experiment_name = os.path.basename(os.path.dirname(experiment_dir))

            # Find the latest checkpoint for each seed in the experiment
            latest_checkpoints = find_latest_checkpoint(experiment_dir)

            for checkpoint in latest_checkpoints:
                destination_blob_name = f"{bucket_name}/checkpoints/{experiment_name}/{os.path.basename(os.path.dirname(checkpoint))}/{os.path.basename(checkpoint)}"
                script.write(
                    f'gsutil -m cp -n "{checkpoint}" "gs://{destination_blob_name}"\n'
                )

    print(f"Written to {shell_script_file}")
