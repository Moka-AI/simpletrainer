from __future__ import annotations

from typing import Type

from simpletrainer.loggers.base import DeepLearningLogger
from simpletrainer.loggers.mlflow import MlflowLogger
from simpletrainer.loggers.tensorboard import TensorboardLogger
from simpletrainer.loggers.wandb import WandbLogger

DeepLearningLoggerRegistry: dict[str, Type[DeepLearningLogger]] = {
    'tensorboard': TensorboardLogger,
    'wandb': WandbLogger,
    'mlflow': MlflowLogger,
}
