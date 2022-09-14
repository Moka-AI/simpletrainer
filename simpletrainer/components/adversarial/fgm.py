# -*- coding: utf-8 -*-
import re

import torch

from simpletrainer import AttrsComponent, Trainer, after, define, field
from simpletrainer.utils.torch import get_params_with_pattern


@define(tags=('adversarial',))
class FGM(AttrsComponent):
    pattern: re.Pattern = field(converter=re.compile, hyper_param=True)
    adv_weight: float = field(default=1.0, hyper_param=True)
    eps: float = field(default=1e-8, hyper_param=True)

    @after(Trainer.set_loss, try_first=True)
    def add_adv(self, trainer: Trainer):
        if trainer.in_train_stage:
            return

        loss = trainer.stage_state.loss
        params = get_params_with_pattern(trainer.model, self.pattern)

        grad_params = torch.autograd.grad(outputs=loss, inputs=params, create_graph=True)
        params_noise = [self.calculate_noise(grad) for grad in grad_params]

        for param, noise in zip(params, params_noise):
            param.data.add_(noise)

        fgm_loss = self.adv_weight * trainer.model(**trainer.stage_state.batch)['loss']
        fgm_loss.backward()
        trainer.stage_state.metrics['fgm_loss'] = fgm_loss.item()

        for param, noise in zip(params, params_noise):
            param.data.sub_(noise)

    def calculate_noise(self, grad: torch.Tensor) -> torch.Tensor:
        return self.adv_weight * (grad / (grad.pow(2).sum() + self.eps))
