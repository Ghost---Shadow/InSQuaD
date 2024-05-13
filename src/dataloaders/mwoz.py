from collections import defaultdict
import json
from dataloaders.base import BaseDataset
from prompt_formatting_strategies.bare import BareStrategy


class MwozDataset(BaseDataset):
    LABELS = None

    NAME = "mwoz"
    OVERRIDE_PROMPT_FORMATTING_STRATEGY = BareStrategy.NAME

    def __init__(self, config):
        super().__init__(config)
        self.cached_load_dataset(MwozDataset.NAME, ("multi_woz_v22", "v2.2"))

    @staticmethod
    def speaker_and_utterance_to_conversation(speakers, utterances):
        unique_speakers = list(set(speakers))
        speaker_name_lut = {i: f"User {i+1}: " for i in unique_speakers}

        conversation = ""
        for speaker, utterance in zip(speakers, utterances):
            conversation += speaker_name_lut[speaker] + utterance + "\n"

        return conversation

    @staticmethod
    def accumulate_slot_values(frames):
        result = defaultdict(set)
        for frame in frames:
            for state in frame["state"]:
                slot_names = state["slots_values"]["slots_values_name"]
                slot_values = state["slots_values"]["slots_values_list"]
                for slot_name, slot_value_list in zip(slot_names, slot_values):
                    result[slot_name].update(set(slot_value_list))

        for key in result:
            result[key] = list(result[key])[0]

        return dict(result)

    @staticmethod
    def collate_fn(batch):
        prompts = []
        labels = []
        for row in batch:
            speakers = row["turns"]["speaker"]
            utterances = row["turns"]["utterance"]
            frames = row["turns"]["frames"]
            prompt = MwozDataset.speaker_and_utterance_to_conversation(
                speakers, utterances
            )
            slot_values = MwozDataset.accumulate_slot_values(frames)

            prompts.append("\n" + prompt + "Answer: ")
            label = json.dumps(slot_values)
            for bad_stuff in ['"', "'", "-", "{", "}"]:
                label = label.replace(bad_stuff, "")
            labels.append(label)

        return {"prompts": prompts, "labels": labels}
