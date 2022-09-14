import logging
from typing import Literal, Optional

from torch.optim.lr_scheduler import _LRScheduler  # type: ignore

from simpletrainer import BaseComponent, Trainer, after
from simpletrainer.integrations import is_transformers_available

logger = logging.getLogger(__name__)
SchedulerTypeStr = Literal[
    'linear',
    'cosine',
    'cosine_with_restarts',
    'polynomial',
    'constant',
    'constant_with_warmup',
]


class TransformersLRScheduler(BaseComponent):
    only_main_process = False
    hyper_param_names = ['lr_scheduler_type', 'warmup_steps', 'max_warmup_steps']
    lr_scheduler: '_LRScheduler'

    def __init__(
        self,
        lr_scheduler_type: SchedulerTypeStr,
        warmup_steps: Optional[float] = None,
        total_steps: Optional[int] = None,
        max_warmup_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not is_transformers_available():
            raise ValueError('transformers is not installed')
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_warmup_steps = max_warmup_steps

    def with_trainer(self, trainer: Trainer) -> None:
        total_steps_from_trainer = trainer.total_steps
        if self.total_steps is None:
            if total_steps_from_trainer is None:
                raise ValueError('Can not infer total steps from lr scheduler')
            else:
                logger.info(
                    'use trainer.total_steps (infer from dataloader), total_steps: %s',
                    total_steps_from_trainer,
                )
                total_steps = total_steps_from_trainer
        else:
            if self.total_steps != total_steps_from_trainer:
                logger.warning(
                    f'self.total_steps ({self.total_steps}) is not equal to \
                    trainer.total_steps ({total_steps_from_trainer})'
                )
            total_steps = int(self.total_steps)

        if self.warmup_steps is not None:
            if self.warmup_steps < 1:
                num_warmup_steps = int(self.warmup_steps * total_steps)
                if self.max_warmup_steps:
                    num_warmup_steps = min(num_warmup_steps, self.max_warmup_steps)
            else:
                num_warmup_steps = int(self.warmup_steps)
        else:
            num_warmup_steps = None
        self.warmup_steps = num_warmup_steps

        from transformers import get_scheduler

        self.lr_scheduler = get_scheduler(  # type: ignore
            name=self.lr_scheduler_type,
            optimizer=trainer.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

    @after(Trainer.step, try_first=True)
    def step(self, trainer: Trainer) -> None:
        self.lr_scheduler.step()
