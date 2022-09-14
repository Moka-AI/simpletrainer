# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from simpletrainer.common.registry import DeepLearningLoggerRegistry
from simpletrainer.common.types import HyperParams, MetricDict

if TYPE_CHECKING:
    from simpletrainer.core.trainer import Trainer


class BaseDeepLearningLogger:
    def __init_subclass__(cls, /, name: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        DeepLearningLoggerRegistry[name] = cls

    def with_trainer(self, trainer: 'Trainer') -> None:
        raise NotImplementedError

    def log_scalars(self, scalars: dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        raise NotImplementedError

    def log_tensors(self, tensors: dict[str, torch.Tensor], step: Optional[int] = None) -> None:
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
