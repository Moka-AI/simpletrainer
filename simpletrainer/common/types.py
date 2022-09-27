from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import torch
import torch.utils.data
from typing_extensions import TypeAlias

Batch: TypeAlias = Mapping[str, Any]
BatchOutput: TypeAlias = Mapping[str, Any]
MetricDict: TypeAlias = Dict[str, float]
Dataset: TypeAlias = torch.utils.data.Dataset[Batch]
LRScheduler: TypeAlias = torch.optim.lr_scheduler._LRScheduler
Prime: TypeAlias = Union[str, float, int, bool, None]
JSON: TypeAlias = Union[Prime, List['JSON'], Dict[str, 'JSON']]
HyperParams: TypeAlias = Dict[str, Prime]
Device: TypeAlias = Union[str, torch.device]
PathOrStr: TypeAlias = Union[str, Path]
