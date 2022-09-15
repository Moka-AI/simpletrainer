import sys
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

_T = TypeVar('_T')
_C = TypeVar('_C', bound=type)

_EqOrderType = Union[bool, Callable[[Any], Any]]
_ValidatorType = Callable[[Any, Attribute[_T], _T], Any]
_ConverterType = Callable[[Any], Any]
_FilterType = Callable[[Attribute[_T], _T], bool]
_ReprType = Callable[[Any], str]
_ReprArgType = Union[bool, _ReprType]
_OnSetAttrType = Callable[[Any, Attribute[Any], Any], Any]
_OnSetAttrArgType = Union[_OnSetAttrType, List[_OnSetAttrType]]
_FieldTransformer = Callable[[type, List[Attribute[Any]]], List[Attribute[Any]]]
# FIXME: in reality, if multiple validators are passed they must be in a list
# or tuple, but those are invariant and so would prevent subtypes of
# _ValidatorType from working when passed in a list or tuple.
_ValidatorArgType = Union[_ValidatorType[_T], Sequence[_ValidatorType[_T]]]

if sys.version_info >= (3, 8):
    from typing import Literal
    @overload
    def Factory(factory: Callable[[], _T]) -> _T: ...
    @overload
    def Factory(
        factory: Callable[[Any], _T],
        takes_self: Literal[True],
    ) -> _T: ...
    @overload
    def Factory(
        factory: Callable[[], _T],
        takes_self: Literal[False],
    ) -> _T: ...

else:
    @overload
    def Factory(factory: Callable[[], _T]) -> _T: ...
    @overload
    def Factory(
        factory: Union[Callable[[Any], _T], Callable[[], _T]],
        takes_self: bool = ...,
    ) -> _T: ...

# Static type inference support via __dataclass_transform__ implemented as per:
# https://github.com/microsoft/pyright/blob/1.1.135/specs/dataclass_transforms.md
# This annotation must be applied to all overloads of "define" and "attrs"
#
# NOTE: This is a typing construct and does not exist at runtime.  Extensions
# wrapping attrs decorators should declare a separate __dataclass_transform__
# signature in the extension module using the specification linked above to
# provide pyright support.
def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
) -> Callable[[_T], _T]: ...

class Attribute(Generic[_T]):
    name: str
    default: Optional[_T]
    validator: Optional[_ValidatorType[_T]]
    repr: _ReprArgType
    cmp: _EqOrderType
    eq: _EqOrderType
    order: _EqOrderType
    hash: Optional[bool]
    init: bool
    converter: Optional[_ConverterType]
    metadata: Dict[Any, Any]
    type: Optional[Type[_T]]
    kw_only: bool
    on_setattr: _OnSetAttrType
    def evolve(self, **changes: Any) -> 'Attribute[Any]': ...

# NOTE: We had several choices for the annotation to use for type arg:
# 1) Type[_T]
#   - Pros: Handles simple cases correctly
#   - Cons: Might produce less informative errors in the case of conflicting
#     TypeVars e.g. `attr.ib(default='bad', type=int)`
# 2) Callable[..., _T]
#   - Pros: Better error messages than #1 for conflicting TypeVars
#   - Cons: Terrible error messages for validator checks.
#   e.g. attr.ib(type=int, validator=validate_str)
#        -> error: Cannot infer function type argument
# 3) type (and do all of the work in the mypy plugin)
#   - Pros: Simple here, and we could customize the plugin with our own errors.
#   - Cons: Would need to write mypy plugin code to handle all the cases.
# We chose option #1.

# `attr` lies about its return type to make the following possible:
#     attr()    -> Any
#     attr(8)   -> int
#     attr(validator=<some callable>)  -> Whatever the callable expects.
# This makes this type of assignments possible:
#     x: int = attr(8)
#
# This form catches explicit None or no default but with no other arguments
# returns Any.
@overload
def attrib(
    default: None = ...,
    validator: None = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: None = ...,
    converter: None = ...,
    factory: None = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> Any: ...

# This form catches an explicit None or no default and infers the type from the
# other arguments.
@overload
def attrib(
    default: None = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: Optional[Type[_T]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> _T: ...

# This form catches an explicit default argument.
@overload
def attrib(
    default: _T,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: Optional[Type[_T]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> _T: ...

# This form covers type=non-Type: e.g. forward references (str), Any
@overload
def attrib(
    default: Optional[_T] = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: object = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> Any: ...
@overload
def field(
    *,
    default: None = ...,
    export: bool = ...,
    hyper_param: bool = ...,
    save: bool = ...,
    validator: None = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: None = ...,
    factory: None = ...,
    kw_only: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> Any: ...

# This form catches an explicit None or no default and infers the type from the
# other arguments.
@overload
def field(
    *,
    default: None = ...,
    export: bool = ...,
    hyper_param: bool = ...,
    save: bool = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> _T: ...

# This form catches an explicit default argument.
@overload
def field(
    *,
    default: _T,
    export: bool = ...,
    hyper_param: bool = ...,
    save: bool = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> _T: ...

# This form covers type=non-Type: e.g. forward references (str), Any
@overload
def field(
    *,
    default: Optional[_T] = ...,
    export: bool = ...,
    hyper_param: bool = ...,
    save: bool = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
) -> Any: ...
@overload
@__dataclass_transform__(order_default=True, field_descriptors=(attrib, field))
def attrs(
    maybe_cls: _C,
    these: Optional[Dict[str, Any]] = ...,
    repr_ns: Optional[str] = ...,
    repr: bool = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    auto_detect: bool = ...,
    collect_by_mro: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> _C: ...
@overload
@__dataclass_transform__(order_default=True, field_descriptors=(attrib, field))
def attrs(
    maybe_cls: None = ...,
    these: Optional[Dict[str, Any]] = ...,
    repr_ns: Optional[str] = ...,
    repr: bool = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    auto_detect: bool = ...,
    collect_by_mro: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> Callable[[_C], _C]: ...
@overload
@__dataclass_transform__(field_descriptors=(attrib, field))
def define(
    maybe_cls: _C,
    *,
    only_main_process: bool = ...,
    tags: Optional[Sequence[str]] = ...,
    these: Optional[Dict[str, Any]] = ...,
    repr: bool = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    auto_detect: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> _C: ...
@overload
@__dataclass_transform__(field_descriptors=(attrib, field))
def define(
    maybe_cls: None = ...,
    *,
    these: Optional[Dict[str, Any]] = ...,
    only_main_process: bool = ...,
    tags: Optional[Sequence[str]] = ...,
    repr: bool = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    auto_detect: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> Callable[[_C], _C]: ...
