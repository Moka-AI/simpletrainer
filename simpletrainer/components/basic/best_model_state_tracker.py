# -*- coding: utf-8 -*-
from typing import Mapping, Optional, Union

import torch

from simpletrainer import AttrsComponent, Trainer, after, define, field
from simpletrainer.common.types import SignMetric
from simpletrainer.components.basic.metric_tracker import MetricTracker


@define(only_main_process=True)
class BestModelStateTracker(AttrsComponent):
    sign_metric: SignMetric = field(default=None, converter=SignMetric.from_str)
    best_model_state: Mapping[str, torch.Tensor] = field(init=False, export=True)
    metric_tracker: MetricTracker = field(init=False)

    def __init__(self, sign_metric: Optional[Union[str, SignMetric]] = None):
        self._init_sign_metric = sign_metric

    def with_trainer(self, trainer: Trainer) -> None:
        sign_metric = self._init_sign_metric or trainer.core_metric
        if isinstance(sign_metric, str):
            sign_metric = SignMetric.from_str(sign_metric)
        self.sign_metric = sign_metric
        self.metric_tracker = trainer.hook_engine.setdefault(MetricTracker(sign_metric=self.sign_metric))

    @after(Trainer.run_epoch)
    def track_best_model(self, trainer: Trainer) -> None:
        if self.metric_tracker.best_epoch == trainer.current_epoch:
            self.best_model_state = trainer.model.state_dict()
