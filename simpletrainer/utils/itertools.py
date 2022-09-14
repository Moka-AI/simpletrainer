from __future__ import annotations

from typing import Callable, Iterable, TypeVar

T = TypeVar('T')


def split(items: Iterable[T], split_fn: Callable[[T], bool]) -> tuple[list[T], list[T]]:
    items_a = []
    items_b = []
    for item in items:
        if split_fn(item):
            items_b.append(item)
        else:
            items_a.append(item)
    return items_a, items_b
