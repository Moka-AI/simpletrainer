from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter

from simpletrainer.common.types.hints import HyperParams, MetricDict
from simpletrainer.loggers.base import BaseDeepLearningLogger

if TYPE_CHECKING:
    from simpletrainer.core import Trainer


class TensorboardLogger(BaseDeepLearningLogger, name='tensorboard'):
    writer: SummaryWriter

    def with_trainer(self, trainer: 'Trainer') -> None:
        self.writer = SummaryWriter(str(trainer.output_dir))

    def log_scalars(self, scalars: dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        for name, scalar in scalars.items():
            self.writer.add_scalar(name, scalar, step)

    def log_tensors(self, tensors: dict[str, torch.Tensor], step: Optional[int] = None) -> None:
        for name, tensor in tensors.items():
            tensor = tensor.cpu().data.numpy().flatten()
            self.writer.add_histogram(name, tensor, step)

    def log_summary(self, hyper_params: HyperParams, metrics: MetricDict) -> None:
        metric_dict = {f'hparam/{k}': v for k, v in metrics.items()}
        exp, ssi, sei = hparams(hyper_params, metric_dict)
        self.writer.file_writer.add_summary(exp)  # type: ignore
        self.writer.file_writer.add_summary(ssi)  # type: ignore
        self.writer.file_writer.add_summary(sei)  # type: ignore
        for k, v in metric_dict.items():
            self.writer.add_scalar(k, v)

    def log_artifact(self, artifact: str) -> None:
        warnings.warn('Tensorboard does not support artifact logging')

    def teardown(self) -> None:
        self.writer.close()
