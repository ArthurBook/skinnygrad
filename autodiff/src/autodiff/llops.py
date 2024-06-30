"""
These are the low level ops that the backend must support.
Tribute to [Chief Keef - Opps](https://www.youtube.com/watch?v=0XbrR1veyyI)
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import TYPE_CHECKING, Any, Generic, Iterator, Literal, Sequence, TypeVar, overload

from autodiff import config, llops

if TYPE_CHECKING:
    from autodiff import runtime

RefType = TypeVar("RefType")
PyArrayRepr = int | float | bool | Sequence["PyArrayRepr"]


@dataclasses.dataclass(slots=True)
class Symbol(Generic[RefType]):
    """
    Vertex and its inbound edges in the computational graph
    post execution, materialize as `NOOP` with `buffer` holding the result
    """

    op: LLOps
    src: tuple[Symbol[RefType], ...]
    shape: Shape
    args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
    _buffer: runtime.Buffer[RefType] | None = dataclasses.field(default=None)

    def realize(self) -> runtime.Buffer[RefType]:
        if (buffer := self.buffer) is None:
            self.buffer = buffer = config.Configuration.engine.run(self)
        return buffer

    @property
    def buffer(self) -> runtime.Buffer[RefType] | None:
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: runtime.Buffer[RefType]) -> None:
        if self._buffer is not None:
            assert self._buffer == buffer, f"Buffer ref is immutable {self._buffer=}<-{self.buffer}"
            return
        self.op, self.src, self.args, self._buffer = llops.ControlOps.REALIZED, (), (), buffer


@dataclasses.dataclass(slots=True, frozen=True)
class Shape(Sequence[int]):
    dims: tuple[int, ...]

    def __getitem__(self, index: int) -> int:
        return self.dims[index]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Shape) and self.dims == other.dims

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Shape({', '.join(map(str, self.dims))})"

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self) -> Iterator[int]:
        return iter(self.dims)

    def broadcast(self, other: Shape) -> Shape:
        if self == other:
            return self
        if len(self) > len(other):
            return self.broadcast(other.lpad(len(self) - len(other)))
        if len(self) < len(other):
            return other.broadcast(self.lpad(len(other) - len(self)))
        for a, b in zip(self, other):
            assert a == 1 or b == 1 or a == b, f"Cannot broadcast between {self=} <--> {other=}"
        return Shape(tuple(max(a, b) for a, b in zip(self, other)))

    def dropaxis(self, axis: int) -> Shape:
        return Shape(tuple(d for i, d in enumerate(self) if i != axis))

    def lpad(self, n: int) -> Shape:
        return Shape((1,) * n + self.dims)

    def flat(self) -> Shape:
        return Shape((self.size,))

    @property
    def size(self) -> int:
        return math.prod(self.dims)

    @classmethod
    def from_data(cls, data: PyArrayRepr, /) -> Shape:
        return (
            cls(tuple((len(data), *cls.from_data(data[0]))))
            if isinstance(data, Sequence)
            else cls(())
        )


@enum.global_enum
class ControlOps(enum.Enum):
    LOAD = enum.auto()  # create the tensor from an init instruction
    RESHAPE = enum.auto()  # reshape the tensor
    EXPAND = enum.auto()  # broadcast the tensor
    ASSIGN = enum.auto()  # elementwise assign from one tensor to another (shapes must match)
    REALIZED = enum.auto()  # do nothing. used for realized vertices

    @overload
    def __call__(self: Literal[ControlOps.LOAD], data: PyArrayRepr, /) -> Symbol: ...
    @overload
    def __call__(self: Literal[ControlOps.RESHAPE], src: Symbol, shape: Shape, /) -> Symbol: ...
    @overload
    def __call__(self: Literal[ControlOps.EXPAND], src: Symbol, shape: Shape, /) -> Symbol: ...
    @overload
    def __call__(self: Literal[ControlOps.ASSIGN], target: Symbol, src: Symbol, /) -> Symbol: ...
    def __call__(self, *args) -> Symbol:
        if self is ControlOps.LOAD:
            assert isinstance(src := args[0], (int, float, bool, Sequence)), f"Unknown {src=}"
            return Symbol(self, src=(), args=(src,), shape=Shape.from_data(src))
        if self is ControlOps.RESHAPE:
            assert isinstance(src := args[0], Symbol), f"{src=} is not a Symbol"
            assert isinstance(shape := args[1], Shape), f"{shape=} is not a Shape"
            assert len(src.shape) == len(shape), f"{src.shape=} != {shape=}"
            return Symbol(self, src=(src,), args=(shape.dims,), shape=shape)
        if self is ControlOps.EXPAND:
            assert isinstance(src := args[0], Symbol), f"{src=} is not a Symbol"
            assert isinstance(shape := args[1], Shape), f"{shape=} is not a Shape"
            return Symbol(self, src=(src,), args=(shape.dims,), shape=shape)
        if self is ControlOps.ASSIGN:
            assert isinstance(src := args[0], Symbol), f"{src=} is not a Symbol"
            assert isinstance(target := args[1], Symbol), f"{target=} is not a Symbol"
            assert_shape_match(src.shape, target.shape)
            return Symbol(self, src=args, shape=src.shape)
        if self is ControlOps.REALIZED:
            raise RuntimeError(f"{self=} should not be called")
        raise NotImplementedError(f"{self=} is not implemented")


@enum.global_enum
class UnaryOps(enum.Enum):  # elementwise apply f(a:M)->b:M
    NEG = enum.auto()  # turn the value negative

    def __call__(self, src: Symbol) -> Symbol:
        return Symbol(self, src=(src,), shape=src.shape)


@enum.global_enum
class BinaryOps(enum.StrEnum):  # elementwise apply f(a:A,b:A)->c:A
    ADD = enum.auto()  # addition a+b
    MUL = enum.auto()  # multiplication a*b

    def __call__(self, src1: Symbol, src2: Symbol) -> Symbol:
        assert_shape_match(src1.shape, src2.shape)
        return Symbol(self, src=(src1, src2), shape=src1.shape)


@enum.global_enum
class TernaryOps(enum.Enum):  # elementwise apply f(a:A,b:A,c:A)->d:A
    WHERE = enum.auto()  # where a, take b, else c

    def __call__(self, src1: Symbol, src2: Symbol, src3: Symbol) -> Symbol:
        assert_shape_match(src1.shape, src2.shape, src3.shape)
        return Symbol(self, src=(src1, src2, src3), shape=src1.shape)


@enum.global_enum
class ReduceOps(enum.Enum):  # reduce a along axis=int f(a:A)->b:B

    SUM = enum.auto()  # sum along axis
    MAX = enum.auto()  # max along axis

    def __call__(self, src: Symbol, axis: int) -> Symbol:
        assert isinstance(axis, int), f"{axis=} is not an int"
        return Symbol(self, src=(src,), args=(axis,), shape=src.shape.dropaxis(axis))


## These are all the ops that the backend must support
LLOps = ControlOps | UnaryOps | BinaryOps | TernaryOps | ReduceOps


###
# Helpers
###
def assert_shape_match(*shapes: Shape) -> None:
    assert all(shapes[0] == shape for shape in shapes[1:]), f"{shapes=} do not match"
