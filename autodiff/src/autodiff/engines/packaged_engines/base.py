""" 
generic packaged backend
packaged backends work by mapping the autodiff LLOPs to
a set of callables
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Generic, Literal, Protocol, TypeVar, overload

from autodiff import graph, llops, runtime

ObjRefType = TypeVar("ObjRefType")
logger = logging.getLogger(__name__)


class PackagedEngine(runtime.Engine[ObjRefType]):
    def __init__(self, backend: MappedBackend[ObjRefType]) -> None:
        self._ops_map = backend
        self._execution_lock = threading.Lock()
        self._runtime_cache = weakref.WeakValueDictionary[int, ObjRefType]()

    def run(self, symbol: graph.Symbol[ObjRefType]) -> runtime.Buffer[ObjRefType]:
        if (realized_buffer := symbol.buffer) is None:
            realized_refs = (src_buf.realize().objref for src_buf in symbol.src)
            op = self._ops_map.get(symbol.op)  # type: ignore | TODO FIX
            realized_buffer = runtime.Buffer(op(*realized_refs, *symbol.args), self)
        return realized_buffer

    def to_python(self, array: ObjRefType) -> graph.PyArrayRepresentation:
        return self._ops_map.get(llops.MemOps.TO_PYTHON)(array)


### Backend mapping from llops to callables ###
class MappedBackend(Protocol, Generic[ObjRefType]):
    @overload
    def get(self, op: Literal[llops.MemOps.TO_PYTHON]) -> ToPython[ObjRefType]: ...
    @overload
    def get(self, op: Literal[llops.MemOps.INIT]) -> Load[ObjRefType]: ...
    @overload
    def get(self, op: Literal[llops.MemOps.ASSIGN]) -> Assign[ObjRefType]: ...
    @overload
    def get(self, op: llops.UnaryOps) -> UnaryOp[ObjRefType]: ...
    @overload
    def get(self, op: llops.BinaryOps) -> BinaryOp[ObjRefType]: ...
    @overload
    def get(self, op: llops.TernaryOps) -> TernaryOp[ObjRefType]: ...
    @overload
    def get(self, op: llops.ReduceOps) -> ReduceOp[ObjRefType]: ...


### Signatures for the backend refs ###
class ToPython(Protocol, Generic[ObjRefType]):  # type: ignore | ObjRefType variance is irrelevant
    def __call__(self, ref: ObjRefType, /) -> graph.PyArrayRepresentation: ...
class Load(Protocol, Generic[ObjRefType]):  # type: ignore | ObjRefType variance is irrelevant
    def __call__(self, pyobj: graph.PyArrayRepresentation, /) -> ObjRefType: ...
class Assign(Protocol, Generic[ObjRefType]):  # type: ignore | ObjRefType variance is irrelevant
    def __call__(self, target: ObjRefType, src: ObjRefType, /) -> None: ...
class UnaryOp(Protocol, Generic[ObjRefType]):
    def __call__(self, ref: ObjRefType, /) -> ObjRefType: ...
class BinaryOp(Protocol, Generic[ObjRefType]):
    def __call__(self, ref1: ObjRefType, ref2: ObjRefType, /) -> ObjRefType: ...
class TernaryOp(Protocol, Generic[ObjRefType]):
    def __call__(self, ref1: ObjRefType, ref2: ObjRefType, ref3: ObjRefType, /) -> ObjRefType: ...
class ReduceOp(Protocol, Generic[ObjRefType]):
    def __call__(self, ref: ObjRefType, axis: int, /) -> ObjRefType: ...
