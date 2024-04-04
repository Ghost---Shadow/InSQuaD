from dataloaders import DATALOADERS_LUT
from dense_indexes import DENSE_INDEXES_LUT
from generative_models import GENERATIVE_MODELS_LUT
from losses import LOSSES_LUT
from prompt_formatting_strategies import PROMPT_FORMATTING_STRATEGIES_LUT
from semantic_search_models import SEMANTIC_SEARCH_MODELS_LUT
from subset_selection_strategies import SUBSET_SELECTION_STRATEGIES_LUT
from training_strategies import TRAINING_STRATEGIES_LUT
import yaml
from pydantic import BaseModel, validator
from pathlib import Path
from typing import List


def type_validator(lut):
    """A factory function that creates a validator function for the 'type' field,
    using the provided Look-Up Table (LUT)."""

    def validate_type_field(cls, value):
        if value not in lut:
            raise ValueError(
                f"{value} is not a valid type. Must be one of {list(lut.keys())}."
            )
        return value

    return validate_type_field


class WandBConfig(BaseModel):
    project: str
    name: str
    entity: str = None


class GenerativeModelConfig(BaseModel):
    type: str
    # TODO: NoOp does not need checkpoint
    checkpoint: str
    device: str

    _validate_type = validator("type", allow_reuse=True)(
        type_validator(GENERATIVE_MODELS_LUT)
    )


class SemanticSearchModelConfig(BaseModel):
    type: str
    checkpoint: str
    device: str

    _validate_type = validator("type", allow_reuse=True)(
        type_validator(SEMANTIC_SEARCH_MODELS_LUT)
    )


class SubsetSelectionStrategyConfig(BaseModel):
    type: str
    # TODO: Check if either is present
    k: int = None
    gain_cutoff: float = None

    _validate_type = validator("type", allow_reuse=True)(
        type_validator(SUBSET_SELECTION_STRATEGIES_LUT)
    )


class DenseIndexConfig(BaseModel):
    type: str
    index_class: str
    repopulate_every: str
    k_for_rerank: int

    _validate_type = validator("type", allow_reuse=True)(
        type_validator(DENSE_INDEXES_LUT)
    )


class PromptFormattingStrategyConfig(BaseModel):
    type: str

    _validate_type = validator("type", allow_reuse=True)(
        type_validator(PROMPT_FORMATTING_STRATEGIES_LUT)
    )


class ArchitectureConfig(BaseModel):
    generative_model: GenerativeModelConfig
    semantic_search_model: SemanticSearchModelConfig
    subset_selection_strategy: SubsetSelectionStrategyConfig
    dense_index: DenseIndexConfig
    prompt_formatting_strategy: PromptFormattingStrategyConfig


class TrainingLossConfig(BaseModel):
    type: str
    lambd: float = 0.0
    _validate_type = validator("type", allow_reuse=True)(type_validator(LOSSES_LUT))


class TrainingConfig(BaseModel):
    type: str
    dataset: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    learning_rate_decay_strategy: str
    warmup_ratio: float
    seeds: List[int]
    loss: TrainingLossConfig
    _validate_type = validator("type", allow_reuse=True)(
        type_validator(TRAINING_STRATEGIES_LUT)
    )
    _validate_dataset = validator("dataset", allow_reuse=True)(
        type_validator(DATALOADERS_LUT)
    )


class ValidationConfig(BaseModel):
    type: str
    q_d_tradeoff_lambda: float
    annotation_budget: int
    datasets: List[str]
    generative_model: GenerativeModelConfig

    _validate_datasets = validator("datasets", each_item=True, allow_reuse=True)(
        type_validator(DATALOADERS_LUT)
    )


class RootConfig(BaseModel):
    wandb: WandBConfig
    architecture: ArchitectureConfig
    validation: ValidationConfig
    training: TrainingConfig


class Config:
    @classmethod
    def from_file(cls, file_name):
        filepath = Path(file_name).resolve()
        config_data = cls._load_from_file(filepath)
        return cls._create_from_dict(config_data, filepath)

    @classmethod
    def from_dict(cls, some_dict):
        config_data = some_dict
        return cls._create_from_dict(config_data)

    @staticmethod
    def _load_from_file(filepath):
        with open(filepath, "r") as file:
            config_data = yaml.safe_load(file)
        return config_data

    @staticmethod
    def _create_from_dict(config_data, filepath=None):
        # Instantiate RootConfig with the loaded data
        root_config = RootConfig(**config_data)

        # Perform the filename vs wandb.name validation if filepath is provided
        if filepath:
            basename = filepath.stem
            if basename != root_config.wandb.name:
                raise ValueError(
                    f"Configuration filename '{basename}' does not match wandb.name '{root_config.wandb.name}'"
                )

        return root_config
