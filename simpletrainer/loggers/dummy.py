from __future__ import annotations

from typing import Optional, Union

import torch

from simpletrainer.common.types import HyperParams, MetricDict
from simpletrainer.loggers.base import DeepLearningLogger


class DummyLogger(DeepLearningLogger):
    def log_metrics(self, scalars: dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        pass

    def log_tensors(self, tensors: dict[str, torch.Tensor], step: Optional[int] = None) -> None:
        pass

    def log_summary(self, hyper_params: HyperParams, metrics: MetricDict) -> None:
        pass

    def log_artifact(self, artifact: str) -> None:
        pass
