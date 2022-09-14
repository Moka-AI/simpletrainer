# -*- coding: utf-8 -*-
import re
from typing import Optional, Pattern, Sequence

import torch
from torch.cuda.amp import GradScaler, autocast

from simpletrainer import AttrsComponent, Trainer, after, define, field
from simpletrainer.utils.torch import get_params_with_pattern


@define
class GradPenalty(AttrsComponent):
    pattern: Pattern = field(converter=re.compile, hyper_param=True)
    penalty_weight: float = field(default=1.0, hyper_param=True)

    @after(Trainer.set_loss, try_first=True)
    def add_penalty(self, trainer: Trainer):
        if trainer.in_valid_stage:
            return

        loss = trainer.stage_state.loss
        params = get_params_with_pattern(trainer.model, self.pattern)

        grad_scaler: Optional[GradScaler] = trainer.grad_scaler
        if grad_scaler is not None:
            grad_params = torch.autograd.grad(outputs=grad_scaler.scale(loss), inputs=params, create_graph=True)  # type: ignore
            inv_scale = 1.0 / grad_scaler.get_scale()
            grad_params = [p * inv_scale for p in grad_params]
            with autocast():
                penalty = self.calculate_penalty(grad_params)
        else:
            grad_params = torch.autograd.grad(outputs=loss, inputs=params, create_graph=True)
            penalty = self.calculate_penalty(grad_params)
        trainer.update_metrics({'grad_penalty': (penalty * self.penalty_weight).item()})
        trainer.stage_state.loss = loss + penalty * self.penalty_weight

    @staticmethod
    def calculate_penalty(grad_params: Sequence[torch.Tensor]):
        if not grad_params:
            raise ValueError('No parameters found')

        grad_norm = torch.tensor(0, device=grad_params[0].device, dtype=grad_params[0].dtype)
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return grad_norm
