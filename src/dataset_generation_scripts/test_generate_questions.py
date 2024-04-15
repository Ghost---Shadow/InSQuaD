from dataset_generation_scripts.process_hotpotqa import add_question_to_row
from datasets import load_dataset
from generative_models.wrapped_t5 import WrappedT5
import unittest


# python -m unittest dataset_generation_scripts.generate_questions_test.TestAddQuestionToRow -v
@unittest.skip("needs gpu")
class TestAddQuestionToRow(unittest.TestCase):
    # python -m unittest dataset_generation_scripts.generate_questions_test.TestAddQuestionToRow.test_with_t5 -v
    def test_with_t5(self):
        dataset = load_dataset("hotpot_qa", "distractor")
        row = dataset["train"][0]

        config = {
            "architecture": {
                "question_generator_model": {
                    "name": "t5",
                    "size": "base",
                    "device": "cuda:0",
                }
            }
        }
        model = WrappedT5(config)

        modified_row = add_question_to_row(model, row)

        assert modified_row["questions"] is not None
