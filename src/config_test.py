import unittest
from config import Config
from pydantic import ValidationError
from pathlib import Path
import yaml


# python -m unittest config_test.TestConfigLoading -v
class TestConfigLoading(unittest.TestCase):
    def test_all_yaml_configs(self):
        root_dir = Path("experiments")
        yaml_files = root_dir.rglob("*.yaml")

        for yaml_file in yaml_files:
            with self.subTest(yaml_file=yaml_file):
                try:
                    Config.from_file(yaml_file)
                except ValidationError as e:
                    self.fail(
                        f"Failed to load configuration from {yaml_file} due to validation error: {e}"
                    )

    # python -m unittest config_test.TestConfigLoading.test_validator -v
    def test_validator(self):
        with open("experiments/dummy_experiment.yaml", "r") as file:
            config_data = yaml.safe_load(file)

        # Should not crash
        Config.from_dict(config_data)

        config_data["datasets"]["train"] = "does_not_exist"

        # Test that ValueError is raised with the expected message
        with self.assertRaises(ValueError) as context:
            Config.from_dict(config_data)

        # Check if the exception message matches the expected format
        expected_msg = "does_not_exist is not a valid type. Must be one of ['mrpc']. [type=value_error, input_value='does_not_exist', input_type=str]"
        self.assertTrue(expected_msg in str(context.exception))


if __name__ == "__main__":
    unittest.main()
