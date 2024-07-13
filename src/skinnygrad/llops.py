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
    constructor: Callable[P, tuple[shapes.Shape, inspect.BoundArguments]]
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


def construct_load(data: PyArrayRepr, /) -> tuple[shapes.Shape, inspect.BoundArguments]:
    return shapes.Shape.from_data(data), bind(construct_load, data)


def construct_reshape(s: Symbol, /, *, shape: shapes.Shape) -> tuple[shapes.Shape, inspect.BoundArguments]:
    assert s.shape.size == shape.size, f"{s.shape=} <> {shape=} size mismatch"
    return shape, bind(construct_reshape, s, shape=shape)


def construct_broadcast(s: Symbol, /, *, shape: shapes.Shape) -> tuple[shapes.Shape, inspect.BoundArguments]:
    return s.shape.broadcast(shape), bind(construct_broadcast, s, shape=shape)


def construct_permute(s: Symbol, /, *, order: Sequence[int]) -> tuple[shapes.Shape, inspect.BoundArguments]:
    assert len(order) == s.shape.ndims, f"{s.shape=} <> {order=} must have same number of dimensions"
    return s.shape.permute(order), bind(construct_permute, s, order=s.shape.normalize_idxs(*order))


def construct_assign(
    s1: Symbol, s2: Symbol, /, *, loc: slice | bool = True
) -> tuple[shapes.Shape, inspect.BoundArguments]:
    assert_shape_match(s1, s2)  # TODO: loc
    return s1.shape, bind(construct_assign, s1, s2, loc=loc)


def construct_unary(s: Symbol, /) -> tuple[shapes.Shape, inspect.BoundArguments]:
    return s.shape, bind(construct_unary, s)


def construct_binary(s1: Symbol, s2: Symbol, /) -> tuple[shapes.Shape, inspect.BoundArguments]:
    assert_shape_match(s1, s2)
    return s1.shape, bind(construct_binary, s1, s2)


def construct_reduce(s: Symbol, /, axes: tuple[int, ...]) -> tuple[shapes.Shape, inspect.BoundArguments]:
    assert isinstance(s, Symbol), f"{s=} must be symbol"
    ndims, normed_axes = s.shape.ndims, sorted(s.shape.normalize_idxs(*axes), reverse=True)
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

    LOAD = Op(construct_load)
    ASSIGN = Op(construct_assign)
    RESHAPE = Op(construct_reshape)
    BROADCAST = Op(construct_broadcast)
    PERMUTE = Op(construct_permute)
    NEG = Op(construct_unary)
    ADD = Op(construct_binary)
    MUL = Op(construct_binary)
    SUM = Op(construct_reduce)
    MAX = Op(construct_reduce)
