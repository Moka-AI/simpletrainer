from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from simpletrainer.common.oprator import Opration
from simpletrainer.core.component import BaseComponent

if TYPE_CHECKING:
    from simpletrainer.core.trainer import Trainer

logger = logging.getLogger(__name__)

T = TypeVar('T')
CT = TypeVar('CT', bound=BaseComponent)
OnOrBeforeMethod = Callable[[Any, 'Trainer'], None]
AfterMethod = Callable[[Any, 'Trainer'], Optional[Opration]]
EntryPoint = Union[Callable, 'TrainerEvent']


class Preposition(str, Enum):
    before = 'before'
    after = 'after'
    on = 'on'


class TrainerEvent(str, Enum):
    INIT = 'INIT'
    PREPARE = 'PREPARE'
    CRASH = 'CRASH'
    FINISH = 'FINISH'
    TEARDOWN = 'TEARDOWN'


def entrypoint(method):
    @wraps(method)
    def wrapped_method(self: 'Trainer', *args, **kwargs):
        self.hook_engine.entrypoint_stack.append(method.__name__)
        self.hook_engine.before(method)
        raw_return = method(self, *args, **kwargs)
        final_return = self.hook_engine.after(method, raw_return)
        self.hook_engine.entrypoint_stack.pop(-1)
        return final_return

    return wrapped_method


def get_entrypoint_name(entrypoint: EntryPoint) -> str:
    if isinstance(EntryPoint, TrainerEvent):
        return entrypoint.value
    elif isinstance(entrypoint, str):
        return entrypoint
    else:
        return entrypoint.__name__


@runtime_checkable
class TrainerHook(Protocol):
    __qualname__: str
    _preposition: Preposition
    _entrypoint_name: str
    _priority: int

    def __call__(self, trainer: Trainer) -> Optional[Opration]:
        ...


class HookCollection:
    def __init__(self) -> None:
        self.data: dict[Preposition, list[TrainerHook]] = {
            Preposition.before: [],
            Preposition.after: [],
            Preposition.on: [],
        }

    def add(self, method: TrainerHook) -> None:
        self.data[method._preposition].append(method)
        self.data[method._preposition].sort(key=lambda method: method._priority)


class TrainerHookEngine:
    def __init__(self, trainer: 'Trainer') -> None:
        # Map: EntryPoint -> HookCollection
        self.entrypoint_hooks: dict[str, HookCollection] = defaultdict(lambda: HookCollection())
        self.trainer = trainer
        self.entrypoint_stack = []

    def _register(self, component: BaseComponent) -> None:
        hooks = self.inspect_trainer_hook(component)
        if not hooks:
            logger.warning(f'{component} do not have any TrainerHook')

        for hook in hooks:
            self.entrypoint_hooks[hook._entrypoint_name].add(hook)

    def _add(self, component: BaseComponent) -> None:
        logger.debug(f'Add component {component.__class__.__name__}')
        self.trainer.components.append(component)
        self.refresh()

    @staticmethod
    def inspect_trainer_hook(obj: Any) -> list[TrainerHook]:
        trainer_hooks: list[TrainerHook] = []
        for _, member in inspect.getmembers(obj):
            is_method_or_func = inspect.ismethod(member) or inspect.isfunction(member)
            if is_method_or_func and isinstance(member, TrainerHook):
                trainer_hooks.append(member)
        return trainer_hooks

    def add(self, component: BaseComponent) -> None:
        component.with_trainer(self.trainer)

        for other_component in self.trainer.components:
            if component == other_component:
                raise ValueError(f'Component {component} already exists in collection')

        for other_component in self.trainer.components:
            component.check_compatibility(other_component)

        if self.trainer.is_main_process or (not component.only_main_process):
            self._add(component)

    def setdefault(self, component: CT) -> CT:
        component.with_trainer(self.trainer)

        for other_component in self.trainer.components:
            if component == other_component:
                return other_component  # type: ignore

        for other_component in self.trainer.components:
            component.check_compatibility(other_component)

        if self.trainer.is_main_process or (not component.only_main_process):
            self._add(component)

        return component

    def pop(self, component: CT) -> CT:
        output_component = self.trainer.components.pop(self.trainer.components.index(component))
        self.refresh()
        return output_component  # type: ignore

    def find(self, component_class: Type[CT], **attrs) -> Optional[CT]:
        for component in self.trainer.components:
            if not isinstance(component, component_class):
                continue

            if all(getattr(component, k) == v for k, v in attrs.items()):
                return component

    def refresh(self) -> None:
        self.entrypoint_hooks = defaultdict(lambda: HookCollection())
        for component in self.trainer.components:
            self._register(component)

    def after(self, entrypoint: EntryPoint, raw_return: T) -> T:
        entrypoint_name = get_entrypoint_name(entrypoint)
        if entrypoint_name not in self.entrypoint_hooks:
            return raw_return

        for hook in self.entrypoint_hooks[entrypoint_name].data[Preposition.after]:
            try:
                opration = hook(self.trainer)
                if opration is not None:
                    new_return = opration(raw_return)
                    assert type(new_return) == type(raw_return)
                    raw_return = new_return
            except Exception as e:
                e.args = (f'{hook.__qualname__}', *e.args)
                raise
        return raw_return

    def before(self, entrypoint: EntryPoint) -> None:
        entrypoint_name = get_entrypoint_name(entrypoint)
        if entrypoint_name not in self.entrypoint_hooks:
            return

        for hook in self.entrypoint_hooks[entrypoint_name].data[Preposition.before]:
            try:
                hook(self.trainer)
            except Exception as e:
                e.args = (f'{hook.__qualname__}', *e.args)
                raise

    def on(self, entrypoint: EntryPoint) -> None:
        entrypoint_name = get_entrypoint_name(entrypoint)
        if entrypoint_name not in self.entrypoint_hooks:
            return

        for hook in self.entrypoint_hooks[entrypoint_name].data[Preposition.on]:
            try:
                hook(self.trainer)
            except Exception as e:
                e.args = (f'{hook.__qualname__}', *e.args)
                raise


def _get_priority(try_first: bool = False, try_last: bool = False) -> int:
    if try_first and try_last:
        raise ValueError("Can't try first and last at the same time")
    priority = 50
    if try_first:
        priority = 0
    if try_last:
        priority = 100
    return priority


def after(
    entrypoint: EntryPoint,
    try_first: bool = False,
    try_last: bool = False,
    priority: Optional[int] = None,
) -> Callable[[AfterMethod], TrainerHook]:
    entrypoint_name = get_entrypoint_name(entrypoint)

    if priority is None:
        priority = _get_priority(try_first, try_last)
    elif priority < 0 or priority > 100:
        raise ValueError('Priority must be between 0 and 100')

    def bounded_target(target: AfterMethod) -> TrainerHook:
        target._entrypoint_name = entrypoint_name
        target._preposition = Preposition.after
        target._priority = priority
        return target  # type: ignore

    return bounded_target


def before(
    entrypoint: EntryPoint,
    try_first: bool = False,
    try_last: bool = False,
    priority: Optional[int] = None,
) -> Callable[[OnOrBeforeMethod], TrainerHook]:
    entrypoint_name = get_entrypoint_name(entrypoint)

    if priority is None:
        priority = _get_priority(try_first, try_last)
    elif priority < 0 or priority > 100:
        raise ValueError('Priority must be between 0 and 100')

    def bounded_target(target: OnOrBeforeMethod) -> TrainerHook:
        target._entrypoint_name = entrypoint_name
        target._preposition = Preposition.before
        target._priority = priority
        return target  # type: ignore

    return bounded_target


def on(
    entrypoint: TrainerEvent,
    try_first: bool = False,
    try_last: bool = False,
    priority: Optional[int] = None,
) -> Callable[[OnOrBeforeMethod], TrainerHook]:
    if priority is None:
        priority = _get_priority(try_first, try_last)
    elif priority < 0 or priority > 100:
        raise ValueError('Priority must be between 0 and 100')

    def bounded_target(target: OnOrBeforeMethod) -> TrainerHook:
        target._entrypoint_name = entrypoint.value
        target._preposition = Preposition.on
        target._priority = priority
        return target  # type: ignore

    return bounded_target
