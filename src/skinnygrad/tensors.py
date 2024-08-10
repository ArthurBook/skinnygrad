"""
Tensors with torch-like interface
"""

from __future__ import annotations
from typing import Self

from skinnygrad import autograd
import numpy as np


class Tensor(autograd.AutoDiffable):
    # shaping
    reshape = autograd.reshape
    repeat = autograd.repeat
    broadcast = autograd.broadcast
    flatten = autograd.flatten
    permute = autograd.permute
    transpose = autograd.transpose
    # movement
    __getitem__ = autograd.select
    pool = autograd.pool
    pad = autograd.pad
    # arithmetic
    __add__ = __radd__ = autograd.add
    __sub__ = __rsub__ = autograd.sub
    __mul__ = __rmul__ = autograd.mul
    __truediv__ = __rtruediv__ = autograd.div
    __matmul__ = __rmatmul__ = autograd.matmul
    reciprocal = autograd.reciprocal
    sum = autograd.sum
    max = autograd.amax
    conv = autograd.conv
    # activations
    relu = autograd.relu
    sigmoid = autograd.sigmoid
    softmax = autograd.softmax

    # constructors
    @classmethod
    def random_uniform(cls, *shape: int, lb: float = 0, ub: float = 1) -> Self:
        return cls(np.random.uniform(lb, ub, shape).tolist())

    @classmethod
    def random_normal(cls, *shape: int, mean: float = 0, var: float = 1) -> Self:
        return cls(np.random.normal(mean, var, shape).tolist())  # TODO create from random uniform
