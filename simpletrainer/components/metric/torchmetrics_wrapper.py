# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional, Union

from simpletrainer import BaseComponent, DefaultSettings, Trainer, after, on
from simpletrainer.common.types import Batch, BatchOutput, MetricDict
from simpletrainer.integrations import is_torchmetrics_available
from simpletrainer.utils.common import pretty_str, smartgetter

if is_torchmetrics_available():
    import torchmetrics as tm

    TorchMetric = Union[tm.Metric, tm.MetricCollection]


TargetGetter = Callable[[Batch], Any]
PredGetter = Callable[[BatchOutput], Any]


class TorchMetricsWrapper(BaseComponent):
    only_main_process = False

    def __init__(
        self,
        metric: 'TorchMetric',
        on_train: bool = True,
        on_valid: bool = True,
        pred_getter: Optional[PredGetter] = None,
        target_getter: Optional[TargetGetter] = None,
        compute_intervel: Optional[int] = None,
    ):
        if not is_torchmetrics_available():
            raise ImportError('torchmetrics is not available')

        if not on_train and not on_valid:
            raise ValueError('on_train and on_valid must be True at least one')

        if on_train:
            self.train_metric = metric.clone()
        if on_valid:
            self.valid_metric = metric.clone()

        self.on_train = on_train
        self.on_valid = on_valid

        self.pred_getter = pred_getter or smartgetter(DefaultSettings.pred_key)
        self.target_getter = target_getter or smartgetter(DefaultSettings.target_key)
        self.compute_intervel = compute_intervel

    def with_trainer(self, trainer: Trainer) -> None:
        if self.compute_intervel is None:
            self.compute_intervel = self.suggest_intervel(trainer.num_steps_per_epoch)

    @on(Trainer.EVENT.PREPARE)
    def set_device(self, trainer: Trainer):
        if self.on_train:
            self.train_metric.to(trainer.device)
        if self.on_valid:
            self.valid_metric.to(trainer.device)

    @after(Trainer.run_train_batch)
    def update_train_metrics(self, trainer: Trainer):
        if self.on_train:
            self._update(trainer, self.train_metric)

    @after(Trainer.run_valid_batch)
    def update_valid_metrics(self, trainer: Trainer):
        if self.on_valid:
            self._update(trainer, self.train_metric)

    def _update(self, trainer: Trainer, metric: 'TorchMetric'):
        batch = trainer.stage_state.batch
        batch_output = trainer.stage_state.batch_output
        pred = self.pred_getter(batch_output)
        target = self.target_getter(batch)
        metric.update(pred, target)
        if (trainer.stage_state.num_steps % self.compute_intervel) == 0:   # type: ignore
            trainer.update_metrics(self.compute(metric))

    @after(Trainer.run_stage)
    def reset_metrics(self, trainer: Trainer):
        if self.on_train:
            self.train_metric.reset()
        if self.on_valid:
            self.valid_metric.reset()

    @staticmethod
    def compute(metric: 'TorchMetric') -> MetricDict:
        score_dict = metric.compute()
        if not isinstance(score_dict, dict):
            score_dict = {metric.__class__.__name__: score_dict}
        score_dict = {k: v.item() for k, v in score_dict.items()}
        return score_dict

    def suggest_intervel(self, steps_per_epoch: Optional[int]) -> int:
        if steps_per_epoch is None:
            return 100
        else:
            return min((steps_per_epoch // 100) + 1, 100)

    def __repr__(self) -> str:
        fields = {}
        if self.on_train:
            fields['metric'] = repr(self.train_metric)
        else:
            fields['metric'] = repr(self.valid_metric)
        fields.update(
            on_train=self.on_train,
            on_valid=self.on_valid,
            compute_intervel=self.compute_intervel,
        )
        return pretty_str(fields, self.__class__.__name__)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False
