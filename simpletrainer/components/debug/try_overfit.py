# -*- coding: utf-8 -*-
import simpletrainer.common.oprator as op
from simpletrainer import BaseComponent, Trainer, after


class TryOverfit(BaseComponent):
    only_main_process = False

    def with_trainer(self, trainer: Trainer):
        self.batch = next(iter(trainer.train_dataloader))

    @after(Trainer.generate_batch, try_first=True)
    def replace_batch(self, trainer: Trainer):
        return op.Opration(op.replace, self.batch)

    def __repr__(self):
        return 'TryOverfit()'
