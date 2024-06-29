"""
These are the low level ops that the backend must support.
Tribute to [Chief Keef - Opps](https://www.youtube.com/watch?v=0XbrR1veyyI)
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any, Generic, Literal, Sequence, TypeVar, overload

from autodiff import config, llops

if TYPE_CHECKING:
    from autodiff import runtime

RefType = TypeVar("RefType")
PyArrayRepresentation = int | float | bool | Sequence["PyArrayRepresentation"]


@dataclasses.dataclass(slots=True)
class Symbol(Generic[RefType]):
    """
    Vertex and its inbound edges in the computational graph
    post execution, materialize as `NOOP` with `buffer` holding the result
    """

    op: LLOps
    src: tuple[Symbol[RefType], ...]
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


class Shape(tuple[int, ...]): ...


@enum.global_enum
class ControlOps(enum.Enum):
    REALIZED = enum.auto()  # do nothing. used for realized vertices
    LOAD = enum.auto()  # create the tensor from an init instruction
    VIEW = enum.auto()  # reshape the tensor
    ASSIGN = enum.auto()  # elementwise assign from one tensor to another (shapes must match)

    @overload
    def __call__(self: Literal[ControlOps.LOAD], load_instruction: Any, /) -> Symbol: ...
    @overload
    def __call__(self: Literal[ControlOps.ASSIGN], target: Symbol, src: Symbol, /) -> Symbol: ...
    @overload
    def __call__(self: Literal[ControlOps.VIEW], src: Symbol, shape: Shape, /) -> Symbol: ...
    def __call__(self, *args) -> Symbol:
        if self is ControlOps.LOAD:
            assert len(args) == 1, f"{self=} takes 1 arg"
            return Symbol(self, src=(), args=(args[0],))
        if self is ControlOps.ASSIGN:
            assert len(args) == 2, f"{self=} takes 2 args"
            assert isinstance(args[0], Symbol), f"{self=} takes 2 args"
            return Symbol(self, src=args)
        if self is ControlOps.VIEW:
            src, shape = args
            return Symbol(self, src=(src,), args=shape)
        if self is ControlOps.REALIZED:
            raise RuntimeError(f"{self=} should not be called")
        raise NotImplementedError(f"{self=} is not implemented")


@enum.global_enum
class UnaryOps(enum.Enum):  # elementwise apply f(a:M)->b:M
    NEG = enum.auto()  # turn the value negative

    def __call__(self, src: Symbol) -> Symbol:
        return Symbol(self, src=(src,))


@enum.global_enum
class BinaryOps(enum.StrEnum):  # elementwise apply f(a:A,b:A)->c:A
    ADD = enum.auto()  # addition a+b
    MUL = enum.auto()  # multiplication a*b

    def __call__(self, src1: Symbol, src2: Symbol) -> Symbol:
        return Symbol(self, src=(src1, src2))


@enum.global_enum
class TernaryOps(enum.Enum):  # elementwise apply f(a:A,b:A,c:A)->d:A
    WHERE = enum.auto()  # where a, take b, else c

    def __call__(self, src1: Symbol, src2: Symbol, src3: Symbol) -> Symbol:
        return Symbol(self, src=(src1, src2, src3))


@enum.global_enum
class ReduceOps(enum.Enum):  # reduce a along axis=int f(a:A)->b:B

    SUM = enum.auto()  # sum along axis
    MAX = enum.auto()  # max along axis

    def __call__(self, src: Symbol, axis: int) -> Symbol:
        return Symbol(self, src=(src,), args=(axis,))


## These are all the ops that the backend must support
LLOps = ControlOps | UnaryOps | BinaryOps | TernaryOps | ReduceOps
