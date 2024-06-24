"""
The computational graph is built as `Symbol` objects.

Before execution, the `Symbol` obj represesents a lazy operation in the compute DAG.
After execution, the `Symbol` materializes to have a `Buffer` holding the result
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Generic, Literal, Sequence, TypeVar, overload

from autodiff import config, llops

if TYPE_CHECKING:
    from autodiff import runtime

ObjRefType = TypeVar("ObjRefType")
PyArrayRepresentation = int | float | bool | Sequence["PyArrayRepresentation"]


@dataclasses.dataclass(slots=True)
class Symbol(Generic[ObjRefType]):
    """
    Vertex and its inbound edges in the computational graph
    post execution, materialize as `NOOP` with `buffer` holding the result
    """

    op: llops.MemOps | llops.UnaryOps | llops.BinaryOps | llops.TernaryOps | llops.ReduceOps
    src: tuple[Symbol[ObjRefType], ...]
    args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
    _buffer: runtime.Buffer[ObjRefType] | None = dataclasses.field(default=None)

    @staticmethod
    def load(value: PyArrayRepresentation) -> Symbol:
        return Symbol(llops.MemOps.INIT, src=(), args=(value,))

    def realize(self) -> runtime.Buffer[ObjRefType]:
        if (buffer := self.buffer) is None:
            self.buffer = buffer = config.Configuration.backend.run(self)
        return buffer

    @overload
    def do(self, __op: Literal[llops.MemOps.ASSIGN], src: Symbol, /) -> Symbol[ObjRefType]:
        """Assign `src` to `self`"""

    @overload
    def do(self, __op: llops.UnaryOps, /) -> Symbol[ObjRefType]:
        """Elementwise apply `__op` to `self`"""

    @overload
    def do(self, __op: llops.BinaryOps, __b1: Symbol, /) -> Symbol[ObjRefType]:
        """Elementwise apply `__op` between `self` and `__b1`"""

    @overload
    def do(self, __op: llops.TernaryOps, __b1: Symbol, __b2: Symbol, /) -> Symbol[ObjRefType]:
        """Elementwise apply `__op` between `self`, `__b1` and `__b2`"""

    @overload
    def do(self, __op: llops.ReduceOps, /, *, axis: int) -> Symbol[ObjRefType]:
        """Reduce `self` along `axis` with `__op`"""

    def do(self, __op, *src: Symbol, **kwargs: Any) -> Symbol[ObjRefType]:
        """Builds the computational graph"""
        return Symbol(__op, src=(self, *src), args=tuple(kwargs.values()))

    @property
    def buffer(self) -> runtime.Buffer[ObjRefType] | None:
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: runtime.Buffer[ObjRefType]) -> None:
        if self._buffer is not None:
            assert self._buffer == buffer, f"Buffer ref is immutable {self._buffer=}<-{self.buffer}"
            return
        self._buffer = buffer
        self.op = llops.MemOps.NOOP
        self.src = ()  # NOTE: allows garbage collector to do its job
        self.args = ()
