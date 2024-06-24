"""
Inspired by [tinygrad](https://github.com/tinygrad/tinygrad) 
and 
"""

from __future__ import annotations

import abc
import dataclasses
import enum
from typing import Any, Generic, Literal, TypeVar, overload

ObjRefType = TypeVar("ObjRefType")
PyArrayRepresentation = int | float | bool | list["PyArrayRepresentation"]
OpCode = TypeVar("OpCode", bound="LLOpCodes")


###
# Engine
###
class Engine(abc.ABC, Generic[ObjRefType]):
    @abc.abstractmethod
    def execute(self, symbol: Symbol) -> ObjRefType:
        """Execute the symbol and return the engine-native"""

    @abc.abstractmethod
    def to_python(self, array: ObjRefType) -> PyArrayRepresentation: ...


###
# Symbols describe the computational graph for the engine
###
@dataclasses.dataclass(slots=True)
class Symbol(Generic[OpCode]):
    """
    Vertex+in edges in the computational graph
    some symbols have args to describe behaviour
    """

    op: OpCode
    src: tuple[Symbol, ...]
    args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)

    @overload
    def apply(self, __op: Literal[MemOps.ASSIGN], src: Symbol, /) -> Symbol[Literal[MemOps.ASSIGN]]:
        """Assign `src` to `self`"""

    @overload
    def apply(self, __op: UnaryOps, /) -> Symbol[UnaryOps]:
        """Elementwise apply `__op` to `self`"""

    @overload
    def apply(self, __op: BinaryOps, __b1: Symbol, /) -> Symbol[BinaryOps]:
        """Elementwise apply `__op` between `self` and `__b1`"""

    @overload
    def apply(self, __op: TernaryOps, __b1: Symbol, __b2: Symbol, /) -> Symbol[TernaryOps]:
        """Elementwise apply `__op` between `self`, `__b1` and `__b2`"""

    @overload
    def apply(self, __op: ReduceOps, /, *, axis: int) -> Symbol[ReduceOps]:
        """Reduce `self` along `axis` with `__op`"""

    def apply(self, __op, *src: Symbol, **kwargs: Any) -> Symbol:
        return Symbol(__op, src=(self, *src), args=tuple(kwargs.values()))

    @staticmethod
    def load(value: PyArrayRepresentation) -> Symbol[Literal[MemOps.COPY]]:
        return Symbol(MemOps.COPY, src=(), args=(value,))


###
# Low level ops that the execution engine supports
# Tribute to [Chief Keef - Opps](https://www.youtube.com/watch?v=0XbrR1veyyI)
###
class LLOpCodes(enum.Enum): ...


class MemOps(LLOpCodes):
    COPY = enum.auto()
    ASSIGN = enum.auto()


class UnaryOps(LLOpCodes):
    NEG = enum.auto()


class BinaryOps(LLOpCodes):
    ADD = enum.auto()
    MUL = enum.auto()


class TernaryOps(LLOpCodes):
    WHERE = enum.auto()


class ReduceOps(LLOpCodes):
    SUM = enum.auto()
    MAX = enum.auto()
