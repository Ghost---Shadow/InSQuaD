import unittest
from config import Config
from pydantic import ValidationError
from pathlib import Path
import yaml


# python -m unittest config_test.TestConfigLoading -v
class TestConfigLoading(unittest.TestCase):
    # python -m unittest config_test.TestConfigLoading.test_all_yaml_configs -v
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
        with open("experiments/tests/quaild_test_experiment.yaml", "r") as file:
            config_data = yaml.safe_load(file)

        # Should not crash
        Config.from_dict(config_data)

        config_data["training"]["dataset"] = "does_not_exist"

        # Test that ValueError is raised with the expected message
        with self.assertRaises(ValueError):
            Config.from_dict(config_data)

    # python -m unittest config_test.TestConfigLoading.test_name_with_hash -v
    def test_name_with_hash(self):
        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")

        expected = "quaild_test_experiment_abfc1"

        assert config.name_with_hash == expected, config.name_with_hash

        config.offline_validation.datasets.append("potato")
        config.offline_validation.seeds.append(43)
        config.training.seeds.append(44)
        config.training.extra_metrics.append("tomato")

        assert config.name_with_hash == expected, config.name_with_hash

        config.offline_validation.type = "cabbage"

        new_expected = "quaild_test_experiment_0f97c"
        assert config.name_with_hash == new_expected, config.name_with_hash


if __name__ == "__main__":
    unittest.main()
