import unittest
import os
from config import Config
from generative_models.wrapped_t5 import WrappedT5


# python -m unittest generative_models.wrapped_t5_test.TestWrappedT5 -v
class TestWrappedT5(unittest.TestCase):

    def test_evaluate(self):
        config_path = os.path.join("experiments", "dummy_experiment.yaml")
        config = Config.from_file(config_path)
        wrapped_t5 = WrappedT5(config)

        prompt = "The quick brown fox"
        label1 = " jumps over the lazy dog"
        label2 = " brick dig hat mat late"

        result1 = wrapped_t5.evaluate(prompt, label1)
        result2 = wrapped_t5.evaluate(prompt, label2)

        # print(result1, result2)

        assert result1 > result2


if __name__ == "__main__":
    unittest.main()
