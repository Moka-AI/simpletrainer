# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, cast

import torch

from simpletrainer import AttrsComponent, Trainer, after, before, define, field, on
from simpletrainer.components.basic.metric_tracker import MetricTracker
from simpletrainer.utils.torch import (
    get_module_gradient_summary,
    get_module_learning_rate_summary,
    get_module_parameter_summary,
)


@define(only_main_process=True)
class ModelWatcher(AttrsComponent):
    interval: int = 0
    watch_metrics: bool = True
    watch_parameter: bool = False
    watch_gradient: bool = False
    watch_parameter_histogram: bool = False
    watch_learning_rate: bool = False
    watch_output_dir: bool = False
    metric_tracker: MetricTracker = field(init=False)

    def with_trainer(self, trainer: Trainer) -> None:
        self.metric_tracker = trainer.hook_engine.setdefault(MetricTracker(trainer.core_metric))
        self.interval = self.interval or self._suggest_intervel(trainer.num_steps_per_epoch)

    def _suggest_intervel(self, steps_per_epoch: Optional[int]) -> int:
        if steps_per_epoch is None:
            return 100
        else:
            if self.watch_parameter or self.watch_gradient or self.watch_parameter_histogram:
                return min((steps_per_epoch // 20) + 1, 100)
            else:
                return min((steps_per_epoch // 100) + 1, 100)

    @before(Trainer.step)
    def _log_step(self, trainer: Trainer):
        self.interval = cast(int, self.interval)

        step = trainer._loop_state.cum_steps
        if step % self.interval != 0:
            return

        if self.watch_metrics:
            trainer.log(trainer.stage_state.metrics)

        if self.watch_learning_rate:
            lr_summary = get_module_learning_rate_summary(trainer.model, trainer.optimizer)
            trainer.log(lr_summary, prefix='learning_rate')

        if self.watch_parameter:
            param_mean, param_std = get_module_parameter_summary(trainer.model)
            trainer.log(param_mean, prefix='parameter_mean')
            trainer.log(param_std, prefix='parameter_std')

        if self.watch_gradient:
            grad_mean, grad_std = get_module_gradient_summary(trainer.model)
            trainer.log(grad_mean, prefix='gradient_mean')
            trainer.log(grad_std, prefix='gradient_std')

        if self.watch_parameter_histogram:
            parameters_to_log: dict[str, torch.Tensor] = {name: param for name, param in trainer.model.named_parameters()}
            trainer.log(parameters_to_log, prefix='parameter_histogram')

    @after(Trainer.run_valid_stage)
    def watch_valid_metrics(self, trainer: Trainer):
        validation_metrics = trainer.valid_metrics_history[-1]
        trainer.log(validation_metrics, prefix='validation', step=trainer.current_epoch)

    @on(Trainer.EVENT.FINISH, try_last=True)
    def watch_output(self, trainer) -> None:
        if self.watch_output_dir:
            trainer.logger.log_artifact(str(trainer.output_dir))

    @on(Trainer.EVENT.FINISH)
    def watch_best_metrics(self, trainer: Trainer):
        if trainer.do_eval and self.metric_tracker.best_score is not None:
            best_metrics = trainer.valid_metrics_history[self.metric_tracker.best_epoch - 1]
        else:
            best_metrics = trainer.train_metrics_history[-1]
        if self.watch_metrics:
            trainer.logger.log_summary(trainer.hyper_params, best_metrics)
