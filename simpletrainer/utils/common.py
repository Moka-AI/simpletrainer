from __future__ import annotations

import inspect
import logging
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Optional, Union

from coolname import generate_slug


@contextmanager
def temp_random_seed(seed: int):
    old_state = random.getstate()
    random.seed(seed)
    yield
    random.setstate(old_state)


def random_experiment_name() -> str:  # type: ignore
    seed = int(time.time() * 1000)
    with temp_random_seed(seed):
        return generate_slug(2)


def remove_dir(directory: Union[Path, str]) -> None:
    directory = Path(directory)

    if not directory.exists():
        return

    shutil.rmtree(directory)


def get_init_params(obj: Any) -> dict[str, Any]:
    if obj.__class__.__init__ == object.__init__:
        return {}

    init_signature = inspect.signature(obj.__class__.__init__)
    param_names = [
        p for p in init_signature.parameters.values() if p.name != 'self' and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]
    param_names = sorted([p.name for p in param_names])
    params = {}
    for name in param_names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            raise AttributeError(
                f'{obj.__class__.__name__} has no attribute {name},maybe attribute name is different from the __init__ signature'
            )
        params[name] = value
    return params


def pretty_str(fields: dict[str, Any], name: Optional[str] = None) -> str:
    name = name or ''
    format_string = name + '('
    if not fields:
        format_string += ')'
    else:
        for k, v in fields.items():
            format_string += '\n'
            format_string += f'    {k}: {v}'
        format_string += '\n)'
    return format_string


def pretty_obj_str(obj: Any, init: bool = False) -> str:
    fields = get_init_params(obj) if init else obj.__dict__
    name = obj.__class__.__name__
    return pretty_str(fields, name)


def monkey_patch_method(obj, method: Callable) -> None:
    setattr(obj, method.__name__, MethodType(method, obj))


def add_file_handler(log_file: Union[str, Path], level: int = logging.INFO):
    file_handler = logging.FileHandler(str(log_file))
    formatter = logging.Formatter('%(asctime)s - %(message)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logging.root.addHandler(file_handler)


def smartget(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, '__getitem__'):
        return obj[name]
    return default


class smartgetter:
    def __init__(self, name: str, default: Any = None) -> None:
        self.name = name
        self.default = default

    def __call__(self, obj: Any) -> Any:
        return smartget(obj, self.name, self.default)
