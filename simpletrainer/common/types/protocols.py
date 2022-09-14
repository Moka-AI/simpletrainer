from __future__ import annotations

from typing import Any, Protocol

__all__ = ['Serializable', 'Builder']


class Serializable(Protocol):
    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]):
        ...


class Builder(Protocol):
    def build(self):
        ...
