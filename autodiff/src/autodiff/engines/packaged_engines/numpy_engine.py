""" numpy backend computational graph """

from __future__ import annotations

from autodiff import llops
from autodiff.engines import exceptions
from autodiff.engines.packaged_engines import base

try:
    import numpy as np
except ImportError as exc:
    raise exceptions.BackendNotFoundError("numpy not installed... run `pip install numpy`") from exc


NUMPY_ENGINE = base.PackagedEngine[np.ndarray](
    {  # type: ignore | mypy gets confused over dict with mixed types
        ### memory ###
        llops.MemOps.TO_PYTHON: lambda x: x.tolist(),
        llops.MemOps.INIT: np.array,
        llops.MemOps.ASSIGN: lambda target, src: np.copyto(target, src) or target,
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
    },
)
