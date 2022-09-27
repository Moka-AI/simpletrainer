from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]):
        ...
