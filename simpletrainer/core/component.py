from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Optional

from attrs import fields

from simpletrainer.common.types import HyperParams

if TYPE_CHECKING:
    from simpletrainer.core.trainer import Trainer


class BaseComponent:
    only_main_process: ClassVar[bool]
    tags: ClassVar[list[str]] = []

    @property
    def hyper_params(self) -> HyperParams:
        return {}

    @property
    def exports(self) -> dict[str, Any]:
        return {}

    def check_compatibility(self, other_component: BaseComponent) -> Optional[NoReturn]:
        return

    def post_init_with_trainer(self, trainer: Trainer) -> None:
        return


class AttrsComponent(BaseComponent):
    @property
    def hyper_params(self) -> HyperParams:
        component_fileds = fields(self.__class__)  # type: ignore
        hyper_param_fields = [f for f in component_fileds if f.metadata.get('simpletrainer_hyper_param', False)]
        return {f.name: getattr(self, f.name) for f in hyper_param_fields}

    @property
    def exports(self) -> dict[str, Any]:
        component_fileds = fields(self.__class__)   # type: ignore
        export_fields = [f for f in component_fileds if f.metadata.get('simpletrainer_export', False)]
        return {f.name: getattr(self, f.name) for f in export_fields}


class StatefulAttrsComponent(AttrsComponent):
    def state_dict(self):
        component_fileds = fields(self.__class__)   # type: ignore
        save_fields = [f for f in component_fileds if f.metadata.get('simpletrainer_save', False)]
        return {f.name: getattr(self, f.name) for f in save_fields}

    def load_state_dict(self, state_dict: dict[str, Any]):
        for name, value in state_dict.items():
            setattr(self, name, value)


class ComponentCheckError(Exception):
    pass
