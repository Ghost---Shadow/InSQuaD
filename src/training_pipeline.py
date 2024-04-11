from checkpoint_manager import CheckpointManager
from dataloaders import DATALOADERS_LUT
from dense_indexes import DENSE_INDEXES_LUT
from extra_metrics import EXTRA_METRICS_LUT
from generative_models import GENERATIVE_MODELS_LUT
from learning_rate_schedulers import LEARNING_RATE_SCHEDULERS_LUT
from losses import LOSSES_LUT
from notifications.logger import wandb_safe_log
from prompt_formatting_strategies import PROMPT_FORMATTING_STRATEGIES_LUT
from semantic_search_models import SEMANTIC_SEARCH_MODELS_LUT
from subset_selection_strategies import SUBSET_SELECTION_STRATEGIES_LUT
import torch
from train_utils import average_dicts
from training_strategies import TRAINING_STRATEGIES_LUT
from config import RootConfig
from torch.cuda.amp import GradScaler
import torch.optim as optim
from tqdm import tqdm


class TrainingPipeline:
    def __init__(self, config: RootConfig):
        self.config = config

        # Leaf level parts
        self._load_parts(config)
        self.current_step = 0
        self.current_epoch = 0
        self.current_seed = None

        # Higher level parts
        lr_scheduler_type = config.training.learning_rate_decay_strategy
        self.lr_scheduler = LEARNING_RATE_SCHEDULERS_LUT[lr_scheduler_type](
            config, self.optimizer, self.wrapped_train_dataset, self.current_step
        )

        training_strategy_type = config.training.type
        self.training_strategy = TRAINING_STRATEGIES_LUT[training_strategy_type](
            config, self
        )

    def _load_parts(self, config: RootConfig):
        # Generative Model
        print("Loading generative model")
        generative_model_type = config.architecture.generative_model.type
        self.generative_model = GENERATIVE_MODELS_LUT[generative_model_type](config)

        # Semantic Search Model
        print("Loading Semantic Search Model")
        semantic_search_model_type = config.architecture.semantic_search_model.type
        self.semantic_search_model = SEMANTIC_SEARCH_MODELS_LUT[
            semantic_search_model_type
        ](config)

        # Subset Selection Strategy
        subset_selection_strategy_type = (
            config.architecture.subset_selection_strategy.type
        )
        self.subset_selection_strategy = SUBSET_SELECTION_STRATEGIES_LUT[
            subset_selection_strategy_type
        ](config, self)

        # Dense Index
        print("Loading Dense Index")
        dense_index_type = config.architecture.dense_index.type
        self.dense_index = DENSE_INDEXES_LUT[dense_index_type](config)

        # Prompt Formatting Strategy
        prompt_formatting_strategy_type = (
            config.architecture.prompt_formatting_strategy.type
        )
        self.prompt_formatting_strategy = PROMPT_FORMATTING_STRATEGIES_LUT[
            prompt_formatting_strategy_type
        ](config)

        # DataLoader for training dataset
        print("Preparing train loader")
        train_dataset_type = config.training.dataset
        self.wrapped_train_dataset = DATALOADERS_LUT[train_dataset_type](config)

        # DataLoaders for validation datasets
        print("Preparing validation loaders")
        # TODO: Proper online validation support
        self.wrapped_validation_datasets = [self.wrapped_train_dataset]
        # validation_dataset_types = config.validation.datasets
        # self.wrapped_validation_datasets = [
        #     DATALOADERS_LUT[dataset_type](config)
        #     for dataset_type in validation_dataset_types
        # ]

        # Loss Function
        loss_function_type = config.training.loss.type
        self.loss_function = LOSSES_LUT[loss_function_type](config)

        # Optimizer
        print("Preparing optimizer")
        self.scaler = GradScaler()
        self.optimizer = optim.AdamW(
            self.semantic_search_model.get_all_trainable_parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        print("Preparing extra metrics")
        self.extra_metrics = []
        for extra_metric_type in self.config.training.extra_metrics:
            self.extra_metrics.append(EXTRA_METRICS_LUT[extra_metric_type](self))

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self)

        # TODO: Why???
        for param_group in self.optimizer.param_groups:
            param_group.setdefault("initial_lr", config.training.learning_rate)

    @torch.no_grad
    def compute_extra_metrics(self, batch):
        all_metrics = {}

        for extra_metric_generator in self.extra_metrics:
            metrics = extra_metric_generator.generate_metric(batch)
            all_metrics = {**all_metrics, **metrics}

        return all_metrics

    def train_one_epoch(self):
        self.training_strategy.before_each_epoch()

        train_loader = self.wrapped_train_dataset.get_loader(split="train")
        dataset_name = self.wrapped_train_dataset.NAME

        pbar = tqdm(train_loader)

        for batch in pbar:
            self.optimizer.zero_grad()

            # Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                loss = self.training_strategy.train_step(batch)
                extra_metrics = {}
                if self.current_step % 100 == 0:
                    extra_metrics = self.compute_extra_metrics(batch)
                metrics = {
                    "train": {dataset_name: {"loss": loss.item(), **extra_metrics}}
                }
                pbar.set_description(f"Loss: {round(loss.item()*10000)/10000}")
                wandb_safe_log(metrics, step=self.current_step)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()

            # Unscales gradients and calls or skips optimizer.step()
            self.scaler.step(self.optimizer)
            self.lr_scheduler.step()

            # Updates the scale for next iteration
            self.scaler.update()

            self.current_step += 1

        self.current_epoch += 1

    def run_online_validation(self):
        metrics = {}

        # Populate FAISS with latest embeddings
        self.training_strategy.before_each_epoch()

        for validation_dataset in self.wrapped_validation_datasets:
            dataset_name = validation_dataset.NAME
            metrics[dataset_name] = {}

            validation_loader = validation_dataset.get_loader(split="validation")

            all_losses = []
            all_extra_metrics = []
            for batch in tqdm(validation_loader):
                with torch.no_grad():
                    loss = self.training_strategy.train_step(batch)
                    all_losses.append(loss)

                extra_metrics = self.compute_extra_metrics(batch)
                all_extra_metrics.append(extra_metrics)

            avg_extra_metrics = average_dicts(all_extra_metrics)

            metrics[dataset_name] = {
                "loss": torch.stack(all_losses).mean().item(),
                **avg_extra_metrics,
            }

        metrics = {"validation": metrics}

        wandb_safe_log(metrics, step=self.current_step)
