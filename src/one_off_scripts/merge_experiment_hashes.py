import os
import shutil
from glob import glob
from config import Config


def move(s_path, d_path, dry_run):
    print(f"Move {s_path} to {d_path}")
    if not dry_run:
        shutil.move(s_path, d_path)


def rmtree(source_dir, dry_run):
    print(f"Removing {source_dir}")
    if not dry_run:
        shutil.rmtree(source_dir)


def merge_directories(source_dir, dest_dir, dry_run):
    """Merge all files and folders from source_dir to dest_dir, handling conflicts."""
    conflicts = []
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for item in os.listdir(source_dir):
        s_path = os.path.join(source_dir, item)
        d_path = os.path.join(dest_dir, item)
        if os.path.isdir(s_path):
            if not os.path.exists(d_path):
                move(s_path, d_path, dry_run)
            else:
                # For directories, recursively merge them
                conflicts.extend(merge_directories(s_path, d_path, dry_run))
        else:
            # If it's a file and already exists, note a conflict
            if os.path.exists(d_path):
                conflicts.append(item)
            else:
                move(s_path, d_path, dry_run)
    return conflicts


def unshashify(name):
    return "_".join(name.split("_")[:-1])


def merge_experiment_hashes(base_path, dry_run):
    """Merge all experiment data from various hash directories into the current hash directory."""
    yaml_files = glob("./experiments/**/*.yaml")
    configs = [Config.from_file(f) for f in yaml_files]

    for config in configs:
        current_name_hash = config.name_with_hash
        # Ensure the current hash directory exists
        current_dir = f"./{base_path}/{current_name_hash}"
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)

        # Get all other hash directories for the same base experiment
        other_dirs = [
            f"./{base_path}/{f}"
            for f in os.listdir(f"./{base_path}")
            if unshashify(f) == config.name and f != current_name_hash
        ]

        # Merge other directories into the current one
        for source_dir in other_dirs:
            if os.path.exists(source_dir):
                conflicts = merge_directories(source_dir, current_dir, dry_run)
                if conflicts:
                    # {', '.join(conflicts)}
                    print(
                        f"Conflicts encountered while merging {source_dir} into {current_name_hash} in {base_path}:\n"
                    )
                # Optionally delete the old directory if no conflicts
                if not conflicts:
                    rmtree(source_dir, dry_run)


def remove_empty_dirs(path):
    """Recursively remove empty directories."""
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                # print(f"Removed empty directory: {dir_path}")


if __name__ == "__main__":
    # Use script with caution, hash is a safety measure
    merge_experiment_hashes("artifacts", dry_run=False)
    remove_empty_dirs("./artifacts")
    remove_empty_dirs("./checkpoints")
