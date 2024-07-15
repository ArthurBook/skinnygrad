"""
Computational graph
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
from typing import TYPE_CHECKING, Callable, Generic, ParamSpec, Self, Sequence, TypeVar

from skinnygrad import config, shapes

if TYPE_CHECKING:
    from skinnygrad import runtime

P, R = ParamSpec("P"), TypeVar("R")
PyArrayRepr = int | float | bool | Sequence["PyArrayRepr"]
OpSignature = tuple[shapes.Shape, inspect.BoundArguments]


@dataclasses.dataclass(slots=True)
class Symbol(Generic[P]):
    """
    A node in the computation graph.
    op:     The operation that produced this symbol.
    shape:  The shape of the symbol.
    args:   The arguments that were passed to the operation.
    """

    op: Op[P]
    shape: shapes.Shape
    args: inspect.BoundArguments
    _buffer_cache: runtime.Buffer | None = None

    def __post_init__(self) -> None:
        config.Configuration.on_symbol_creation(self)

    def __repr__(self) -> str:
        status = "REALIZED" if self.is_realized else "UNREALIZED"
        return f"<{self.__module__}.{self.__class__.__name__}({status} {self.op!r}, shape={self.shape.dims!r})>"

    def realize(self, engine: None | runtime.Engine[runtime.RefType] = None) -> runtime.Buffer[runtime.RefType]:
        if (buffer := self.buffer) is None:
            buffer = self.buffer = (config.Configuration.engine if engine is None else engine).run(self)
        return buffer

    @property
    def is_realized(self) -> bool:
        return self.buffer is not None

    @property
    def symbol_args(self) -> dict[str, Symbol]:
        return {k: arg for k, arg in self.args.arguments.items() if isinstance(arg, Symbol)}

    @property
    def non_symbol_args(self) -> dict[str, Symbol]:
        return {k: arg for k, arg in self.args.arguments.items() if not isinstance(arg, Symbol)}

    @property
    def buffer(self) -> None | runtime.Buffer:
        return self._buffer_cache

    @buffer.setter
    def buffer(self, buffer: runtime.Buffer) -> None:
        self._buffer_cache = buffer
        self.args.arguments.update((k, None) for k in self.symbol_args)  # NOTE: deref for garbage collector


@dataclasses.dataclass(slots=True, unsafe_hash=True)
class Op(Generic[P]):
    constructor: Callable[P, OpSignature]
    name: str = dataclasses.field(init=False)

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> Symbol[P]:
        return Symbol(self, *self.constructor(*args, **kwds))

    def __set_name__(self, _: type[Ops], name: str) -> None:
        self.name = name

    def __get__(self, *_) -> Self:
        return self

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}({self.name})>"

    def __repr__(self) -> str:
        return str(self)


def construct_load(data: PyArrayRepr, /) -> OpSignature:
    return shapes.Shape.from_data(data), bind(construct_load, data)


def construct_slice(s: Symbol, /, *, loc: Sequence[shapes.Loc], _skip_norm: bool = False) -> OpSignature:
    loc = loc if _skip_norm else s.shape.normalize_loc(loc)
    newshape = s.shape.slice(*loc, _skip_norm=True)
    return newshape, bind(construct_slice, s, loc=loc)


def construct_pad(s: Symbol, /, *loc: tuple[int, int], pad_val: float = 0) -> OpSignature:
    assert len(loc) == s.shape.ndims, f"{s.shape=} <> {loc=} must have same number of dimensions"
    assert all(len(l) == 2 for l in loc), f"{loc=} must be a sequence of (left, right) padding values"
    assert all(l[0] >= 0 and l[1] >= 0 for l in loc), f"{loc=} must all be non-negative"
    return s.shape.pad(*loc), bind(construct_pad, s, *loc, pad_val=pad_val)


def construct_reshape(s: Symbol, /, *, shape: shapes.Shape) -> OpSignature:
    assert s.shape.size == shape.size, f"{s.shape=} <> {shape=} size mismatch"
    return shape, bind(construct_reshape, s, shape=shape)


def construct_broadcast(s: Symbol, /, *, shape: shapes.Shape) -> OpSignature:
    return s.shape.broadcast(shape), bind(construct_broadcast, s, shape=shape)


def construct_permute(s: Symbol, /, *, order: Sequence[int]) -> OpSignature:
    assert len(order) == s.shape.ndims, f"{s.shape=} <> {order=} must have same number of dimensions"
    return s.shape.permute(order), bind(construct_permute, s, order=s.shape.normalize_dim_ref(*order))


def construct_assign(s1: Symbol, s2: Symbol, /, *, loc: slice | bool = True) -> OpSignature:
    assert_shape_match(s1, s2)  # TODO: loc
    return s1.shape, bind(construct_assign, s1, s2, loc=loc)


def construct_unary(s: Symbol, /) -> OpSignature:
    return s.shape, bind(construct_unary, s)


def construct_binary(s1: Symbol, s2: Symbol, /) -> OpSignature:
    assert_shape_match(s1, s2)
    return s1.shape, bind(construct_binary, s1, s2)


def construct_ternary(s1: Symbol, s2: Symbol, s3: Symbol, /) -> OpSignature:
    assert_shape_match(s1, s2, s3)
    return s1.shape, bind(construct_ternary, s1, s2, s3)


def construct_reduce(s: Symbol, /, axes: tuple[int, ...]) -> OpSignature:
    assert isinstance(s, Symbol), f"{s=} must be symbol"
    ndims, normed_axes = s.shape.ndims, sorted(s.shape.normalize_dim_ref(*axes), reverse=True)
    assert all(0 <= ax < ndims for ax in ()), f"{axes=} must be in range [0, {ndims=}]"
    return s.shape.dropaxes(*normed_axes), bind(construct_reduce, s, tuple(normed_axes))


### Helpers ###
def bind(f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> inspect.BoundArguments:
    return inspect.signature(f).bind(*args, **kwargs)


def assert_shape_match(*symbols: Symbol) -> None:
    assert all(isinstance(symbol, Symbol) for symbol in symbols), f"{symbols=} must be symbols"
    assert all(symbols[0].shape == symbol.shape for symbol in symbols[1:]), f"{symbols=} do not match"


### Ops ##
@enum.global_enum
class Ops(enum.Enum):
    """Low level ops that define & execute the computational graph"""

    READ = Op(construct_load)
    ASSIGN = Op(construct_assign)
    RESHAPE = Op(construct_reshape)
    BROADCAST = Op(construct_broadcast)
    PERMUTE = Op(construct_permute)
    SELECT = Op(construct_slice)
    PAD = Op(construct_pad)
    INV = Op(construct_unary)
    NEG = Op(construct_unary)
    EXP = Op(construct_unary)
    EQ = Op(construct_binary)
    LESS = Op(construct_binary)
    ADD = Op(construct_binary)
    MUL = Op(construct_binary)
    SUM = Op(construct_reduce)
    AMAX = Op(construct_reduce)
    WHERE = Op(construct_ternary)
