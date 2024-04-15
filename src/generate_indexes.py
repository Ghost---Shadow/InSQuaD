import os
import re
import inspect
import importlib.util

from tqdm import tqdm


def import_module_from_file(full_path_to_file):
    """Dynamically imports a module from a given file path."""
    module_name = os.path.splitext(os.path.basename(full_path_to_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, full_path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_classes_with_name(directory):
    """Extracts classes with a NAME attribute from Python files in a specified directory."""
    python_files = [
        f
        for f in os.listdir(directory)
        if f.endswith(".py") and f not in ["__init__.py"] and not f.endswith("_test.py")
    ]

    class_details = []

    for file_name in python_files:
        module = import_module_from_file(os.path.join(directory, file_name))
        module_name = module.__name__

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_name and hasattr(obj, "NAME"):
                class_details.append((module_name, obj.__name__, obj.NAME))

    # Sorting class details by module name and then by class name
    class_details.sort(key=lambda x: (x[0], x[1]))

    return class_details


def format_imports_and_lut(directory_name: str, class_details):
    """Formats import statements and a lookup table from given class details."""
    imports = set()
    index_dict = []

    for module_name, class_name, class_name_static in class_details:
        formatted_import = f"from {directory_name}.{module_name} import {class_name}"
        imports.add(formatted_import)
        index_dict.append(f"    {class_name}.NAME: {class_name},")

    formatted_imports = "\n".join(sorted(imports))
    formatted_lut = "\n".join(index_dict)

    output = f"{formatted_imports}\n\n\n{directory_name.upper()}_LUT = {{\n{formatted_lut}\n}}"
    return output


if __name__ == "__main__":
    for directory_name in tqdm(os.listdir("src")):
        directory_path = f"src/{directory_name}"
        if not os.path.isdir(directory_path):
            continue

        # directory_name = "dataloaders"
        class_details = extract_classes_with_name(directory_path)
        output = format_imports_and_lut(directory_name, class_details)
        with open(f"{directory_path}/__init__.py", "w") as f:
            f.write(output)
