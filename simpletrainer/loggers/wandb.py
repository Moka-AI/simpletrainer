from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import torch

from simpletrainer.common.types.hints import HyperParams, MetricDict
from simpletrainer.integrations import is_wandb_available
from simpletrainer.loggers.base import BaseDeepLearningLogger

if TYPE_CHECKING:
    from simpletrainer.core.trainer import Trainer

if is_wandb_available():
    import wandb


class WandbLogger(BaseDeepLearningLogger, name='wandb'):
    def __init__(self):
        if not is_wandb_available():
            raise ImportError('wandb is not installed')

    def with_trainer(self, trainer: Trainer) -> None:
        wandb.init(
            project=trainer.experiment_name,
            name=trainer.run_name,
            reinit=True,
        )

    def log_scalars(self, scalars: dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        for name, scalar in scalars.items():
            wandb.log({name: scalar}, step=step)

    def log_tensors(self, tensors: dict[str, torch.Tensor], step: Optional[int] = None) -> None:
        for name, tensor in tensors.items():
            tensor = tensor.cpu().data.numpy().flatten()
            wandb.log({name: wandb.Histogram(tensor)}, step=step)

    def log_summary(self, hyper_params: HyperParams, metrics: MetricDict) -> None:
        wandb.config.update(hyper_params)
        for k, v in metrics.items():
            wandb.run.summary[k] = v  # type: ignore

    def log_artifact(self, artifact: str) -> None:
        wandb.log_artifact(artifact)

    def teardown(self) -> None:
        wandb.finish()
