import logging

from mini.dataset import MiniDataset
from mini.model import MiniModel
from pydantic import BaseModel
from torch.optim import Adam
from torch.utils.data import DataLoader

from simpletrainer import DefaultSettings, Trainer, TrainerConfig
from simpletrainer.components import MetricTracker, SaveCheckpoint
from simpletrainer.utils.torch import set_seed

logging.basicConfig(level=logging.INFO, format=DefaultSettings.log_format)


class MiniDatasetConfig(BaseModel):
    num_training_samples: int = 100000
    num_validation_samples: int = 10000
    train_batch_size: int = 64
    validation_batch_size: int = 64


class MiniExperimentConfig(BaseModel):
    dataset: MiniDatasetConfig
    trainer: TrainerConfig


def main(cfg: MiniExperimentConfig):
    # Dataset
    train_dataset = MiniDataset(cfg.dataset.num_training_samples)
    validation_dataset = MiniDataset(cfg.dataset.num_validation_samples)

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataset.train_batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.dataset.validation_batch_size)

    # Model, Optimizer & LRSchedulerConfig
    model = MiniModel(3)
    optimizer = Adam(model.parameters())

    # SimpleTrainer
    trainer = Trainer.from_config(cfg.trainer)
    trainer.hook_engine.add(SaveCheckpoint())
    trainer.hook_engine.add(MetricTracker())

    trainer.train(
        model,
        optimizer,
        train_dataloader,
        validation_dataloader,
    )


if __name__ == '__main__':
    set_seed(24)
    trainer_config = TrainerConfig()
    trainer_config.experiment_name = 'simpletrainer-mini'
    trainer_config.run_name = 'test-restore'
    trainer_config.inspect = False
    trainer_config.progress_bar = 'rich'

    cfg = MiniExperimentConfig(
        dataset=MiniDatasetConfig(),
        trainer=trainer_config,
    )
    main(cfg)
