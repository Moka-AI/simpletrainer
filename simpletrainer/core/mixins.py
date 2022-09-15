from __future__ import annotations

from typing import Optional, Union

import torch
from accelerate import Accelerator

from simpletrainer.common.types import MetricDict
from simpletrainer.core.state import TrainerState


class AcceleratorMixin:
    accelerator: Accelerator

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def grad_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        return self.accelerator.scaler

    @property
    def mixed_precision(self) -> str:
        return self.accelerator.mixed_precision


class TrainerStateMixin:
    state: TrainerState

    def advance_batch(self):
        if self.state.in_train_stage:
            self.state.cum_train_batches += 1
        else:
            self.state.cum_valid_batches += 1
        self.state.num_batches += 1

    def advance_step(self):
        self.state.cum_steps += 1
        self.state.num_steps += 1

    def advance_epoch(self):
        self.state.current_epoch += 1

    def update_stage_metrics(self, **metric_dict: float) -> None:
        self.state.stage_metrics.update(metric_dict)

    @property
    def current_epoch(self) -> int:
        return self.state.current_epoch

    @property
    def total_train_batches(self) -> int:
        return self.state.cum_train_batches

    @property
    def total_valid_batches(self) -> int:
        return self.state.cum_valid_batches

    @property
    def cum_steps(self) -> int:
        return self.state.cum_steps

    @property
    def total_steps(self) -> int:
        return self.state.cum_steps

    @property
    def train_metrics_history(self) -> list[MetricDict]:
        return self.state.train_metrics_history

    @property
    def valid_metrics_history(self) -> list[MetricDict]:
        return self.state.valid_metrics_history

    @property
    def latest_train_metrics(self) -> Optional[MetricDict]:
        if self.state.train_metrics_history:
            return self.state.train_metrics_history[-1]

    @property
    def latest_valid_metrics(self) -> Optional[MetricDict]:
        if self.state.valid_metrics_history:
            return self.state.valid_metrics_history[-1]

    @property
    def exception(self) -> Optional[Union[Exception, KeyboardInterrupt]]:
        return self.state.exception

    @exception.setter
    def exception(self, value: Union[Exception, KeyboardInterrupt]):
        self.state.exception = value

    @property
    def in_train_stage(self):
        return self.state.in_train_stage

    @property
    def in_valid_stage(self):
        return not self.state.in_train_stage
