"""
Tensors with torch-like interface
"""

from __future__ import annotations
import math
import operator
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
    __neg__ = autograd.neg
    __add__ = __radd__ = autograd.add
    __sub__ = __rsub__ = autograd.sub
    __mul__ = __rmul__ = autograd.mul
    __truediv__ = __rtruediv__ = autograd.div
    __pow__ = autograd.pow
    __matmul__ = __rmatmul__ = autograd.matmul
    reciprocal = autograd.reciprocal
    sum = autograd.sum
    mean = autograd.mean
    max = autograd.amax
    conv = autograd.conv
    exp = autograd.exp
    log = autograd.log
    # activations
    relu = autograd.relu
    sigmoid = autograd.sigmoid
    softmax = autograd.softmax

    # constructors
    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False) -> Self:
        return cls([0] * math.prod(shape), requires_grad=requires_grad).reshape(shape)

    @classmethod
    def random_uniform(cls, *shape: int, lb: float = 0, ub: float = 1, requires_grad: bool = False) -> Self:
        return cls(np.random.uniform(lb, ub, shape).tolist(), requires_grad=requires_grad)

    @classmethod
    def random_normal(cls, *shape: int, mean: float = 0, var: float = 1, requires_grad: bool = False) -> Self:
        # TODO create from box muller transform
        return cls(np.random.normal(mean, var, shape).tolist(), requires_grad=requires_grad)
