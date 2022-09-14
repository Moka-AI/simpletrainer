from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Optional, Union

import torch

from simpletrainer.common.types.hints import HyperParams, MetricDict
from simpletrainer.integrations import is_mlflow_available
from simpletrainer.loggers.base import BaseDeepLearningLogger

if TYPE_CHECKING:
    from simpletrainer.core.trainer import Trainer

if is_mlflow_available():
    import mlflow


class MlflowLogger(BaseDeepLearningLogger, name='mlflow'):
    def __init__(self, tracking_uri: Optional[str] = None):
        if not is_mlflow_available():
            raise ImportError('wandb is not installed')
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', None)

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

    def with_trainer(self, trainer: 'Trainer') -> None:
        mlflow.set_experiment(trainer.experiment_name)

    def log_scalars(self, scalars: dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        _scalars = {k: float(v) for k, v in scalars.items()}
        mlflow.log_metrics(_scalars, step=step)

    def log_tensors(self, tensors: dict[str, torch.Tensor], step: Optional[int] = None) -> None:
        warnings.warn('MlflowLogger does not support tensor logging')

    def log_summary(self, hyper_params: HyperParams, metrics: MetricDict) -> None:
        mlflow.log_params(hyper_params)
        mlflow.log_metrics(metrics)

    def log_artifact(self, artifact: str) -> None:
        mlflow.log_artifact(artifact)
