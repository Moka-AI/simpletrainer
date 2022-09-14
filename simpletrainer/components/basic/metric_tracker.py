from typing import Optional

from simpletrainer import AttrsComponent, Trainer, after, define, field
from simpletrainer.common.types.sign_metric import SignMetric


@define
class MetricTracker(AttrsComponent):
    sign_metric: SignMetric = field(default=None, converter=SignMetric.from_str)
    train: bool = False
    best_epoch: int = field(init=False)
    best_score: Optional[float] = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.best_epoch = 0
        self.best_score = None

    @after(Trainer.run_stage)
    def track_metric(self, trainer: Trainer):
        if self.train != trainer.in_train_stage:
            return

        if self.train:
            metrics = trainer.latest_train_metrics
        else:
            metrics = trainer.latest_valid_metrics

        if metrics is None:
            raise ValueError(f'Metrics are not available')

        score = metrics.get(self.sign_metric.name)
        if score is None:
            raise ValueError(f'Metric {self.sign_metric.name} not found in {metrics}')

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = trainer.current_epoch
        elif self.sign_metric.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = trainer.current_epoch
