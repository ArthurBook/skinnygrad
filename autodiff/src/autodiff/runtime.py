"""
Engine is the runtime that executes the computational graph
to return a `Buffer` that represents the materialized `Symbol`
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Generic, TypeVar

from autodiff import graph

ObjRefType = TypeVar("ObjRefType")
PyArrayRepresentation = int | float | bool | list["PyArrayRepresentation"]


@dataclasses.dataclass(slots=True)
class Buffer(Generic[ObjRefType]):
    objref: ObjRefType
    engine: Engine[ObjRefType]

    def to_python(self) -> PyArrayRepresentation:
        return self.engine.to_python(self.objref)


class Engine(abc.ABC, Generic[ObjRefType]):
    @abc.abstractmethod
    def run(self, symbol: graph.Symbol[ObjRefType]) -> Buffer[ObjRefType]:
        """Execute the symbol"""

    @abc.abstractmethod
    def to_python(self, objref: ObjRefType) -> PyArrayRepresentation:
        """Return a python representation of the obj ref (meant for debug purposes)"""
