import unittest
from config import Config
from dataloaders.xsum import XsumDataset
from tqdm import tqdm
from train_utils import set_seed


expected_prompt = """
Document: In a nationally televised speech outlining his strategy against IS, he said that any group that threatened America would "find no safe haven".He also announced that 475 US military personnel would be sent to Iraq but said they would not have a combat role.IS controls large parts of Syria and Iraq after a rapid military advance.Its fighters have become notorious for their brutality, beheading enemy soldiers and Western journalists on video.The US has launched over 150 air strikes against the group in Iraq and provided arms to Iraqi and Kurdish forces fighting against IS.Meanwhile, US Secretary of State John Kerry is continuing his Middle East tour, as he tries to build a coalition against IS.He is expected to meet counterparts from key Arab countries and Turkey in Jeddah, Saudi Arabia. A US official was quoted as saying Mr Kerry would discuss military co-operation to facilitate US air strikes.In a 15-minute speech shown at peak time in the US, President Obama vowed that America would lead "a broad coalition to roll back" IS."Working with the Iraqi government, we will expand our efforts beyond protecting our own people and humanitarian missions, so that we're hitting Isil [Islamic State in Iraq and the Levant - the previous name for IS] targets as Iraqi forces go on the offense" he said.He said he would welcome congressional approval for the fight against IS but said that he had the authority to act without it.President Obama was elected in part because of fervent opposition to the 2003 invasion of Iraq, and presided over the US troop pullout from the country.Analysis: Jon Sopel, BBC North America editor, WashingtonFor the first time, Islamic State targets on the ground in Syria will be in the crosshairs of American pilots. The president told the American people: "I have made it clear that we will hunt down terrorists who threaten our country, wherever they are." But he was equally emphatic that the combat on the ground would happen without US troops. Instead the US will ramp up its military assistance to the Syrian opposition.But the president was also at pains to express what this wasn't. "We will not get dragged into another ground war," he insisted. He said that America would lead a broad coalition to roll back this terrorist threat and would not be acting alone. There were two cautionary notes - the first on timescale and also that this would not be risk-free to American servicemen and women. Action is going to start: who knows when it will be mission accomplished.The speech Obama had hoped to avoidLast year, President Obama abandoned plans to launch air strikes in Syria against government forces after congressional opposition.In his speech, he ruled out working with President Bashar al-Assad of Syria, despite the fact that his forces were also engaged in fighting IS."In the fight against Isil, we cannot rely on an Assad regime that terrorises its people: a regime that will never regain the legitimacy it has lost" he said.Instead, he said, the US would seek to strengthen the non-IS Syrian opposition, which fights against both IS and President Assad.Syria's Western-backed National Coalition welcomed Mr Obama's plan, and urged Congress to approve it."The Syrian Coalition... stands ready and willing to partner with the international community not only to defeat [IS] but also rid the Syrian people of the tyranny of the Assad regime," its president Hadi al-Bahra said in a statement, quoted by Reuters news agency.However, the BBC's Jim Muir in northern Iraq says the Syrian opposition is fragmented and dominated by Islamists, who may be opposed to IS but are seeking Islamic rule rather than democracy.Last month, Syria offered to help the US fight Islamic State.But Syrian Foreign Minister Walid Muallem warned that the US had to co-ordinate with the Syrian government before launching any air strikes on its territory. "Anything outside this is considered aggression," he said.
Summary:
""".strip()
expected_label = """
US President Barack Obama has said he will not hesitate to take action against Islamic State (IS) militants in Syria as well as Iraq.
""".strip()


# python -m unittest dataloaders.xsum_test.TestXsumDatasetLoader -v
class TestXsumDatasetLoader(unittest.TestCase):
    # python -m unittest dataloaders.xsum_test.TestXsumDatasetLoader.test_xsum_loader -v
    def test_xsum_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        config.training.batch_size = 1

        xsum_dataset = XsumDataset(config)
        train_loader = xsum_dataset.get_loader(split="train")

        train_batch = next(iter(train_loader))
        assert train_batch["prompts"][0] == expected_prompt, (
            "(" + train_batch["prompts"][0] + ")"
        )
        assert train_batch["labels"][0] == expected_label, train_batch["labels"][0]

    # python -m unittest dataloaders.xsum_test.TestXsumDatasetLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # Set seed for deterministic testing
        set_seed(42)

        config = Config.from_file("experiments/tests/quaild_test_experiment.yaml")
        xsum_dataset = XsumDataset(config)

        train_loader = xsum_dataset.get_loader(split="train")
        for batch in tqdm(train_loader):
            assert "prompts" in batch
            assert "labels" in batch

        validation_loader = xsum_dataset.get_loader(split="validation")
        for batch in tqdm(validation_loader):
            assert "prompts" in batch
            assert "labels" in batch


if __name__ == "__main__":
    unittest.main()
