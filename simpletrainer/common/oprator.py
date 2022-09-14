from typing import Any, Callable, TypeVar

T = TypeVar('T')
Oprator = Callable[[T, Any], T]


def replace(value: T, other_value: T) -> T:
    return other_value


def dict_update(value: dict, other_value: dict) -> dict:
    value.update(other_value)
    return value


def or_(value: bool, other_value: bool) -> bool:
    return value or other_value


def and_(value: bool, other_value: bool) -> bool:
    return value and other_value


class Opration:
    def __init__(self, oprator: Oprator, value: Any):
        self.oprator = oprator
        self.value = value

    def __call__(self, value: T) -> T:
        return self.oprator(value, self.value)
