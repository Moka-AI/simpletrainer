from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional, Union

import torch

from simpletrainer.common.types import Batch, MetricDict


@dataclass
class TrainerStageState:
    batch_iterator: Iterator[Batch]
    train: bool
    current_epoch: int
    loss_metric_scale: int
    num_steps: int = 0
    num_batches: int = 0
    _loss: torch.Tensor = torch.tensor(0.0)
    metrics: Dict[str, float] = field(default_factory=dict)
    batch: Any = None
    batch_output: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metrics = {
            'loss': 0.0,
        }

    def __repr__(self) -> str:
        stage = 'train' if self.train else 'valid'
        return f'epoch{self.current_epoch}-{stage}'

    @property
    def loss(self) -> torch.Tensor:
        return self._loss

    @loss.setter
    def loss(self, value: torch.Tensor):
        self._loss = value
        self._update_loss_metric()

    def _update_loss_metric(self):
        loss_scalar = float(self.loss.item()) * self.loss_metric_scale

        if math.isnan(loss_scalar):
            return

        self.metrics['loss'] += (loss_scalar - self.metrics['loss']) / (self.num_batches)


@dataclass
class TrainerLoopState:
    current_epoch: int = 1
    cum_train_batches: int = 0
    cum_valid_batches: int = 0
    cum_steps: int = 0
    train_metrics_history: list[MetricDict] = field(default_factory=list)
    valid_metrics_history: list[MetricDict] = field(default_factory=list)
    exception: Optional[Union[Exception, KeyboardInterrupt]] = None
