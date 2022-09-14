# type: ignore
from typing import Any, Optional, Sequence

from attrs import NOTHING
from attrs import define as attrs_define
from attrs import field as attrs_field


def define(
    maybe_cls=None,
    *,
    these=None,
    only_main_process: bool = False,
    tags: Optional[Sequence[str]] = None,
    repr=None,
    hash=None,
    init=None,
    slots=True,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=None,
    kw_only=False,
    cache_hash=False,
    auto_exc=True,
    eq=None,
    order=False,
    auto_detect=True,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
):
    def new_func(cls):
        cls = attrs_define(
            maybe_cls=cls,
            these=these,
            repr=repr,
            hash=hash,
            init=init,
            slots=slots,
            frozen=frozen,
            weakref_slot=weakref_slot,
            str=str,
            auto_attribs=auto_attribs,
            kw_only=kw_only,
            cache_hash=cache_hash,
            auto_exc=auto_exc,
            eq=eq,
            order=order,
            auto_detect=auto_detect,
            getstate_setstate=getstate_setstate,
            on_setattr=on_setattr,
            field_transformer=field_transformer,
            match_args=match_args,
        )
        setattr(cls, 'only_main_process', only_main_process)
        setattr(cls, 'tags', tags or [])
        return cls

    if maybe_cls is None:
        return new_func
    else:
        return new_func(maybe_cls)


def field(
    *,
    default=NOTHING,
    export: bool = False,
    hyper_param: bool = False,
    validator=None,
    repr=True,
    hash=None,
    init=True,
    metadata=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
) -> Any:
    new_metadata = metadata or {}
    new_metadata.update(simpletrainer_export=export, simpletrainer_hyper_param=hyper_param)
    return attrs_field(
        default=default,
        validator=validator,
        repr=repr,
        hash=hash,
        init=init,
        metadata=new_metadata,
        converter=converter,
        factory=factory,
        kw_only=kw_only,
        eq=eq,
        order=order,
        on_setattr=on_setattr,
    )
