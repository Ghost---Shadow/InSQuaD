import unittest
from generative_models.openai_chat_model import OpenAIChatModel


# python -m unittest generative_models.openai_chat_model_test.TestOpenAiChatModel -v
@unittest.skip("API is no longer deterministic")
class TestOpenAiChatModel(unittest.TestCase):
    # python -m unittest generative_models.openai_chat_model_test.TestOpenAiChatModel.test_generate_question -v
    def test_generate_question(self):
        config = {
            "architecture": {"question_generator_model": {"name": "gpt-3.5-turbo"}}
        }
        model = OpenAIChatModel(config)
        result = model.generate_question("The name of the cat is Toby")
        assert result == "What is the name of the cat in the story?", result

        result = model.generate_question(
            "The name of the fox with red tail is Toby and blue tail is Rob."
        )
        assert (
            result
            == "What are the names of the two foxes based on the color of their tails?"
        ), result

        result = model.generate_question("Tom ate a banana because he was sick")
        assert result == "Why did Tom eat a banana?", result

    # python -m unittest generative_models.openai_chat_model_test.TestOpenAiChatModel.test_generate_question_batch_lossy -v
    def test_generate_question_batch_lossy(self):
        config = {
            "architecture": {"question_generator_model": {"name": "gpt-3.5-turbo"}}
        }
        model = OpenAIChatModel(config)
        result = model.generate_question_batch_lossy(
            [
                "The name of the cat is Toby",
                "The name of the fox with red tail is Toby and blue tail is Rob.",
                "Tom ate a banana because he was sick",
            ]
        )
        assert result == [
            "What is the name of the cat?",
            "What are the names of the foxes with red and blue tails?",
            "Why did Tom eat a banana?",
        ]
