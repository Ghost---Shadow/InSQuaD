import os
from training_strategies.no_operation import NoOperation
import yaml


def generate_shell_scripts(root_dir):
    UPLOAD_ARTIFACTS = "source devops/upload_artifacts.sh\n\n"
    UPLOAD_CHECKPOINTS = "source devops/upload_all_checkpoints.sh\n\n"
    STOP_COMMAND = "\nsource devops/stop_current_gcp_instance.sh\n"
    root_dir = os.path.abspath(root_dir)  # Ensure absolute path
    for dirpath, dirnames, filenames in os.walk(root_dir):
        yaml_files = [f for f in filenames if f.endswith(".yaml")]
        if yaml_files:
            train_script_path = os.path.join(dirpath, "_run_train.sh")
            eval_script_path = os.path.join(dirpath, "_run_eval.sh")

            with open(train_script_path, "w") as train_script, open(
                eval_script_path, "w"
            ) as eval_script:
                for yaml_file in yaml_files:
                    config_path = os.path.join(dirpath, yaml_file)
                    # Make config_path relative to root_dir and replace backslashes if on a non-Unix system
                    config_path = os.path.relpath(config_path, start="./").replace(
                        "\\", "/"
                    )
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    if config["training"]["type"] != NoOperation.NAME:
                        train_command = f"python src/train.py --config={config_path}\n"
                    else:
                        train_command = None
                    eval_command = (
                        f"python src/offline_eval.py --config={config_path}\n"
                    )
                    if train_command:
                        train_script.write(train_command)
                        train_script.write(UPLOAD_CHECKPOINTS)
                    eval_script.write(eval_command)
                    eval_script.write(UPLOAD_ARTIFACTS)
                train_script.write(STOP_COMMAND)
                eval_script.write(STOP_COMMAND)

            print(
                f"Generated {train_script_path} and {eval_script_path} for YAML files in {dirpath}"
            )


if __name__ == "__main__":
    generate_shell_scripts("./experiments")
