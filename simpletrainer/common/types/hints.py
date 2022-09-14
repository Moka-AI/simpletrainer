from __future__ import annotations

from typing import Any, Dict, List, Mapping, Union

import torch
import torch.utils.data

__all__ = ['Batch', 'BatchOutput', 'MetricDict', 'Dataset', 'PyTorchScheduler', 'Prime', 'JSON', 'HyperParams', 'Device']

Batch = Mapping[str, Any]
BatchOutput = Mapping[str, Any]
MetricDict = Dict[str, float]
Dataset = torch.utils.data.Dataset[Batch]
PyTorchScheduler = torch.optim.lr_scheduler._LRScheduler
Prime = Union[str, float, int, bool, None]
JSON = Union[Prime, List['JSON'], Dict[str, 'JSON']]
HyperParams = Dict[str, Prime]
Device = Union[str, torch.device]
