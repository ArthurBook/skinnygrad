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
        llops.Ops.SELECT: lambda arr, loc: arr[*(i if isinstance(i, int) else slice(*i) for i in loc)],
        llops.Ops.RESHAPE: lambda src, newshape: np.reshape(src, newshape.dims),
        llops.Ops.BROADCAST: lambda src, newshape: np.broadcast_to(src, newshape.dims),
        llops.Ops.PERMUTE: np.transpose,
        llops.Ops.ASSIGN: lambda tgt, src, i: (np.copyto(tgt, src, where=i), tgt)[1],
        llops.Ops.PAD: lambda src, pads, pad_val: np.pad(src, pad_width=pads, constant_values=(pad_val,)),
        llops.Ops.EXP: np.exp,
        llops.Ops.NEG: np.negative,
        llops.Ops.INV: np.reciprocal,
        llops.Ops.ADD: np.add,
        llops.Ops.MUL: np.multiply,
        llops.Ops.SUM: np.sum,
    }

    def execute(self, op: llops.Op, *args: np.ndarray | Any) -> np.ndarray:
        return self.__OPS_MAP__[op](*args)

    def to_python(self, objref: np.ndarray) -> PyArrayRepresentation:
        return objref.tolist()
