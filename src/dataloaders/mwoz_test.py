import json
import unittest
from config import Config
from dataloaders.mwoz import MwozDataset
from tqdm import tqdm
from train_utils import set_seed


# python -m unittest dataloaders.mwoz_test.TestMwozDatasetLoader -v
class TestMwozDatasetLoader(unittest.TestCase):
    # python -m unittest dataloaders.mwoz_test.TestMwozDatasetLoader.test_mwoz_loader -v
    def test_mwoz_loader(self):
        # Set seed for deterministic testing
        set_seed(4)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        dataset = MwozDataset(config)
        train_loader = dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        expected_batch = {
            "prompts": [
                "User 1: Hi. I need help finding a hotel. Can you help me?\nUser 2: Yes, I can! First, do you have a particular area of town you are interested in?\nUser 1: No and it doesn't need internet or free parking. I would prefer a moderate price though.\nUser 2: In that price range, I have 3 different options for you. All are hotels. I have two in the north and one in the centre of town. Which do you prefer?\nUser 1: Does one of those options come with free wifi?\nUser 2: The ashley hotel and the lovell lodge provides free wifi.\nUser 1: Lets try booking the Ashley Hotel. I need it for 2 people starting Monday for 3 nights please.\nUser 2: I was able to book your party of two into the Ashley Hotel for 3 nights starting on Monday. Your reference is X21XYR7K . Is there anything else I can help with?\nUser 1: I also need to book a train. I need to depart from stevenage and go to cambridge.\nUser 2: What day and time will you be leaving?\nUser 1: I will be leaving on Monday, anytime. The train will need to be booked for 2 people also.\nUser 2: I have train TR9175 leaving at 5:54 and arriving at 6:43. Would youlike to make reservations?\nUser 1: Yes please book it.\nUser 2: I booked it for you. Your reference number is V93OC1AL .\nUser 1: Great, how much is that ticket please?\nUser 2: That will be 12.80 pounds. Is there anything else that I can do for you?\nUser 1: No thank you. That was all I needed. Good bye.\nUser 2: You're welcome. Have a nice day.\n\nAnswer: "
            ],
            "labels": [
                "hoteltype: hotel, hotelarea: dontcare, hotelinternet: no, hotelparking: no, hotelpricerange: moderate, hotelbookday: monday, hotelbookpeople: 2, hotelbookstay: 3, hotelname: ashley hotel, traindeparture: stevenage, traindestination: cambridge, trainbookpeople: 2, trainday: monday"
            ],
        }

        # with open("src/dataloaders/mwoz_sample.json", "w") as f:
        #     json.dump(train_batch, f, indent=2)

        assert train_batch == expected_batch, train_batch

    # python -m unittest dataloaders.mwoz_test.TestMwozDatasetLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        dataset = MwozDataset(config)

        train_loader = dataset.get_loader(split="train")
        for batch in tqdm(train_loader, desc="train"):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader, desc="validation"):
            assert "prompts" in batch
            assert "labels" in batch


if __name__ == "__main__":
    unittest.main()
