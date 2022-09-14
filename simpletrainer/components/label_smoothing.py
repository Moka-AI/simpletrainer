# -*- coding: utf-8 -*-
import inspect
import logging
from typing import Callable

import torch

from simpletrainer import AttrsComponent, Trainer, define, field, on

logger = logging.getLogger(__name__)
CrossEntropyGetter = Callable[[Trainer], torch.nn.CrossEntropyLoss]


def auto_find_cross_entropy(trainer: Trainer):
    for _, v in inspect.getmembers(trainer.model):
        if isinstance(v, torch.nn.CrossEntropyLoss):
            return v
    raise ValueError('No CrossEntropyLoss found in model')


@define
class LabelSmoothing(AttrsComponent):
    rate: float = field(default=0.1, hyper_param=True)
    cross_entropy_getter: CrossEntropyGetter = auto_find_cross_entropy

    @on(Trainer.EVENT.PREPARE)
    def set_label_smoothing(self, trainer: Trainer):
        try:
            loss_instance = self.cross_entropy_getter(trainer)
            loss_instance.label_smoothing = self.rate
            logging.info(
                'get cross entropy loss success, set label smoothing {}',
                self.rate,
            )
        except Exception:
            logging.warning('get cross entropy loss failed, LabelSmoothing disabled')
