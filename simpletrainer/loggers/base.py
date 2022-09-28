# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from simpletrainer.common.types import HyperParams, MetricDict

if TYPE_CHECKING:
    from simpletrainer.core.trainer import Trainer


class DeepLearningLogger(ABC):
    name: ClassVar[str]

    def post_init(self, trainer: 'Trainer') -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        raise NotImplementedError

    def log_tensors(self, tensors: dict[str, torch.Tensor], step: int | None = None) -> None:
        raise NotImplementedError

    def log_summary(self, hyper_params: HyperParams, metrics: MetricDict) -> None:
        raise NotImplementedError

    def log_artifact(self, artifact: str) -> None:
        raise NotImplementedError

    @staticmethod
    def add_prefix(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
        return {f'{prefix}/{key}': value for key, value in data.items()}

    def teardown(self) -> None:
        pass
