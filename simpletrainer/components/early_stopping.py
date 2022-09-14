# -*- coding: utf-8 -*-
import logging
from typing import Optional, Union

from simpletrainer import AttrsComponent, Trainer, after, define, field, op
from simpletrainer.common.types import SignMetric
from simpletrainer.components.basic import MetricTracker

logger = logging.getLogger(__name__)


@define
class EarlyStopping(AttrsComponent):
    patience: int = field(default=1, hyper_param=True)
    sign_metric: SignMetric = field(default=None, converter=SignMetric.from_str)
    metric_tracker: MetricTracker = field(init=False)

    def __init__(
        self,
        patience: int = 1,
        sign_metric: Optional[Union[str, SignMetric]] = None,
    ) -> None:
        if patience <= 0:
            raise ValueError('patience must be greater than 0')

        self.patience = patience
        self._init_sign_metric = sign_metric

    def with_trainer(self, trainer: Trainer) -> None:
        sign_metric = self._init_sign_metric or trainer.core_metric
        if isinstance(sign_metric, str):
            sign_metric = SignMetric.from_str(sign_metric)
        self.sign_metric = sign_metric
        logger.info(f'component EarlyStopping track {self._init_sign_metric} with patience {self.patience}')
        self.metric_tracker = trainer.hook_engine.setdefault(MetricTracker(self.sign_metric))

    @after(Trainer.should_stop_loop)
    def early_stop(self, trainer: Trainer):
        no_patience = (trainer.current_epoch - self.metric_tracker.best_epoch) > self.patience
        if no_patience:
            logger.info(
                f'early stop because no patience, best epoch {self.metric_tracker.best_epoch}, current epoch {trainer.current_epoch}'
            )
        return op.Opration(op.or_, no_patience)
