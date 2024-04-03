from subset_selection_strategies.base_strategy import BaseSubsetSelectionStrategy
import unittest


# python -m unittest subset_selection_strategies.base_strategy_test.TestBaseSubsetSelectionStrategy -v
class TestBaseSubsetSelectionStrategy(unittest.TestCase):
    def test_apply_indexes(self):
        # Input data for testing
        retrieved_documents = [
            {
                "prompts": [
                    "What is Alice's favourite fruit?",
                    "What is Bob's favourite fruit?",
                    "What is Charlie's favourite fruit?",
                ],
                "labels": ["apple", "banana", "coconut"],
                "distances": [0.1, 0.2, 0.3],
            },
            {
                "prompts": [
                    "What is Charlie's favourite fruit?",
                    "What is Bob's favourite fruit?",
                    "What is Alice's favourite fruit?",
                ],
                "labels": ["coconut", "banana", "apple"],
                "distances": [0.1, 0.2, 0.3],
            },
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Alice's favourite fruit?",
                    "What is Charlie's favourite fruit?",
                ],
                "labels": ["banana", "apple", "coconut"],
                "distances": [0.1, 0.2, 0.3],
            },
        ]
        indexes = [
            [1, 2],
            [0, 1],
            [0, 2],
        ]

        # Expected output
        expected_output = [
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Charlie's favourite fruit?",
                ],
                "labels": ["banana", "coconut"],
                "distances": [0.2, 0.3],
            },
            {
                "prompts": [
                    "What is Charlie's favourite fruit?",
                    "What is Bob's favourite fruit?",
                ],
                "labels": ["coconut", "banana"],
                "distances": [0.1, 0.2],
            },
            {
                "prompts": [
                    "What is Bob's favourite fruit?",
                    "What is Charlie's favourite fruit?",
                ],
                "labels": ["banana", "coconut"],
                "distances": [0.1, 0.3],
            },
        ]

        # Create an instance of the strategy
        strategy = BaseSubsetSelectionStrategy()

        # Apply the indexes
        result = strategy.apply_indexes(retrieved_documents, indexes)

        # Assert that the result matches the expected output
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
