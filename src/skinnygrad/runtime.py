"""
Engine is the runtime that executes the computational graph
to return a `Buffer` that represents the materialized `Symbol`
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Generic, TypeVar

import numpy as np

from skinnygrad import llops

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
    def run(self, symbol: llops.Symbol) -> Buffer[RefType]:
        """Execute the symbol"""

    @abc.abstractmethod
    def to_python(self, objref: RefType) -> PyArrayRepresentation:
        """Return a python representation of the obj ref (meant for debug purposes)"""


class SequentialEngine(Engine[RefType], abc.ABC):
    """Executes the graph sequentially in the order of the graph's topological sort"""

    @abc.abstractmethod
    def execute(self, op: llops.Op, *args: RefType | Any) -> RefType: ...

    def run(self, symbol: llops.Symbol) -> Buffer:
        if (realized_buffer := symbol.buffer) is None:
            boundargs = symbol.args.arguments.values()
            args = (v.realize(self).objref if isinstance(v, llops.Symbol) else v for v in boundargs)
            realized_buffer = Buffer(self.execute(symbol.op, *args), self)
        return realized_buffer


### Numpy as default engine implementation ###
class NumPyEngine(SequentialEngine[np.ndarray]):
    __OPS_MAP__ = {
        llops.Ops.LOAD: np.array,
        llops.Ops.ASSIGN: lambda tgt, src, i: (np.copyto(tgt, src, where=i), tgt)[1],
        llops.Ops.RESHAPE: np.reshape,
        llops.Ops.BROADCAST: np.broadcast_to,
        llops.Ops.PERMUTE: np.transpose,
        llops.Ops.NEG: np.negative,
        llops.Ops.ADD: np.add,
        llops.Ops.MUL: np.multiply,
        llops.Ops.SUM: np.sum,
        llops.Ops.MAX: np.amax,
    }

    def execute(self, op: llops.Op, *args: np.ndarray | Any) -> np.ndarray:
        return self.__OPS_MAP__[op](*args)

    def to_python(self, objref: np.ndarray) -> PyArrayRepresentation:
        return objref.tolist()
