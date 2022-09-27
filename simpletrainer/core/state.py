from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import torch
from attrs import define, field, fields

from simpletrainer.common.types import Batch, MetricDict, PathOrStr


def _init_metrics():
    return {'loss': 0.0}


@define
class TrainerState:
    # LoopState
    current_epoch: int = 1
    cum_train_batches: int = 0
    cum_valid_batches: int = 0
    cum_steps: int = 0
    train_metrics_history: list[MetricDict] = field(factory=list)
    valid_metrics_history: list[MetricDict] = field(factory=list)
    entrypoint_stack: list[str] = field(factory=list)
    exception: Optional[Union[Exception, KeyboardInterrupt]] = None

    # StageState
    num_steps: int = 0
    num_batches: int = 0
    stage_metrics: dict[str, float] = field(factory=_init_metrics)
    in_train_stage: bool = True

    batch_iterator: Iterator[Batch] = field(init=False)
    loss: torch.Tensor = field(init=False)
    batch: Batch = field(init=False)
    batch_output: Any = field(init=False)

    # For Restore
    checkpoint_entrypoint_stack: list[str] = field(factory=list)

    @property
    def should_restore(self) -> bool:
        return bool(self.checkpoint_entrypoint_stack)

    def set_loss(self, loss: torch.Tensor, accumulate_grad_batches: int) -> None:
        self.loss = loss / accumulate_grad_batches
        loss_item = loss.item()
        if math.isnan(loss_item):
            return
        self.stage_metrics['loss'] += (loss_item - self.stage_metrics['loss']) / (self.num_batches)

    def prepare(self, train: bool, batch_iterator: Iterator[Batch]):
        self.reset_stage_state()
        self.in_train_stage = train
        self.batch_iterator = batch_iterator

    def restore(self, train: bool, batch_iterator: Iterator[Batch]):
        if train:
            self.batch_iterator = islice(batch_iterator, self.num_batches, None)
        else:
            self.batch_iterator = batch_iterator
        self.checkpoint_entrypoint_stack = []

    def reset_stage_state(self):
        self.num_steps = 0
        self.num_batches = 0
        self.stage_metrics = _init_metrics()

    @classmethod
    def from_json(cls, json_file: PathOrStr) -> TrainerState:
        return cls(**json.loads(Path(json_file).read_text()))

    def to_json(self, json_file: PathOrStr) -> None:
        init_fields = [field.name for field in fields(self.__class__) if field.init]   # type: ignore
        Path(json_file).write_text(json.dumps({field: getattr(self, field) for field in init_fields}, default=str))
