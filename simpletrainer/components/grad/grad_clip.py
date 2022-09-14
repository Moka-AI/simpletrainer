# -*- coding: utf-8 -*-
from typing import Optional

from simpletrainer import AttrsComponent, Trainer, before, define, field


@define
class GradClip(AttrsComponent):
    grad_norm: Optional[float] = field(default=1.0, hyper_param=True)
    grad_value: Optional[float] = field(default=None, hyper_param=True)
    only_main_process = False

    def __attrs_post_init__(self):
        if self.grad_norm is None and self.grad_value is None:
            raise ValueError('Either grad_norm or grad_value must be specified.')

    @before(Trainer.step, try_last=True)
    def clip_grad(self, trainer: Trainer):
        if self.grad_norm is not None:
            trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), self.grad_norm)

        if self.grad_value is not None:
            trainer.accelerator.clip_grad_value_(trainer.model.parameters(), self.grad_value)
