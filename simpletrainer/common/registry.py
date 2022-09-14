from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from simpletrainer.core.loggers.base import BaseDeepLearningLogger


DeepLearningLoggerRegistry: dict[str, Type['BaseDeepLearningLogger']] = {}
