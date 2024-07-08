"""
Engine is the runtime that executes the computational graph
to return a `Buffer` that represents the materialized `Symbol`
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Generic, TypeVar

import numpy as np

from autodiff import llops

RefType = TypeVar("RefType")
PyArrayRepresentation = int | float | bool | list["PyArrayRepresentation"]


@dataclasses.dataclass(slots=True)
class Buffer(Generic[RefType]):
    objref: RefType
    engine: Engine[RefType]

    def to_python(self) -> PyArrayRepresentation:
        return self.engine.to_python(self.objref)


class Engine(abc.ABC, Generic[RefType]):
    """The runtime that executes the computational graph"""

    @abc.abstractmethod
    def run(self, symbol: llops.Symbol[RefType]) -> Buffer[RefType]:
        """Execute the symbol"""

    @abc.abstractmethod
    def to_python(self, objref: RefType) -> PyArrayRepresentation:
        """Return a python representation of the obj ref (meant for debug purposes)"""


class SequentialEngine(Engine[RefType], abc.ABC):
    """Executes the graph sequentially in the order of the graph's topological sort"""

    @abc.abstractmethod
    def execute(self, op: llops.LLOps, *args: RefType | Any) -> RefType: ...

    def run(self, symbol: llops.Symbol[RefType]) -> Buffer[RefType]:
        if (realized_buffer := symbol.buffer) is None:
            refs = (src_buf.realize().objref for src_buf in symbol.src)
            realized_buffer = Buffer(self.execute(symbol.op, *refs, *symbol.args), self)
        return realized_buffer


### Numpy as default engine implementation ###
class NumPyEngine(SequentialEngine[np.ndarray]):
    __OPS_MAP__ = {
        ### control ###
        llops.ControlOps.LOAD: np.array,
        llops.ControlOps.ASSIGN: lambda target, src: (np.copyto(target, src), target)[1],
        llops.ControlOps.RESHAPE: np.reshape,
        llops.ControlOps.EXPAND: np.broadcast_to,
        llops.ControlOps.PERMUTE: np.transpose,
        ### unary ###
        llops.UnaryOps.NEG: np.negative,
        ### binary ###
        llops.BinaryOps.ADD: np.add,
        llops.BinaryOps.MUL: np.multiply,
        ### ternary ###
        llops.TernaryOps.WHERE: np.where,
        ### reduce ###
        llops.ReduceOps.SUM: np.sum,
        llops.ReduceOps.MAX: np.amax,
    }

    def execute(self, op: llops.LLOps, *args: np.ndarray | Any) -> np.ndarray:
        assert op is not llops.ControlOps.REALIZED, f"{op=} can not be executed"
        return self.__OPS_MAP__[op](*args)

    def to_python(self, objref: np.ndarray) -> PyArrayRepresentation:
        return objref.tolist()
