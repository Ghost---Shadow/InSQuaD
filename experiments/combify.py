from ruamel.yaml import YAML
import os


def process_yaml_files(directory, before, after):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096

    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        if before in filename and filename.endswith(".yaml"):
            # Construct the full path to the file
            filepath = os.path.join(directory, filename)

            # Read the YAML file
            with open(filepath, "r") as file:
                data = yaml.load(file)

            # Modify the data according to the specified rules
            data["architecture"]["subset_selection_strategy"]["k"] = data.get(
                "offline_validation", {}
            ).get("annotation_budget", None)
            if "gain_cutoff" in data["architecture"]["subset_selection_strategy"]:
                del data["architecture"]["subset_selection_strategy"]["gain_cutoff"]

            data["offline_validation"]["type"] = "insquad_combinatorial"

            data["wandb"]["name"] = data["wandb"]["name"].replace(before, after)

            # Construct new filename by replacing _gain_ with _comb_
            new_filename = filename.replace(before, after)
            new_filepath = os.path.join(directory, new_filename)

            # Write the modified data back to a new YAML file
            with open(new_filepath, "w") as file:
                yaml.dump(data, file)

            print(f"Processed {filename} into {new_filename}")


if __name__ == "__main__":
    base_path = "./experiments"
    before_after_pairs = [("_gain_", "_comb_"), ("_nt_", "_combnt_")]
    directories = os.listdir(base_path)

    for before, after in before_after_pairs:
        for directory in directories:
            full_path = os.path.join(base_path, directory)
            if os.path.isdir(full_path):
                process_yaml_files(full_path, before, after)
