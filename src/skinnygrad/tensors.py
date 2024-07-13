"""
Tensors with torch-like interface
"""

from __future__ import annotations

from skinnygrad import autograd


class Tensor(autograd.AutoDiffable):
    # shaping
    reshape = autograd.reshape
    broadcast = autograd.broadcast
    flatten = autograd.flatten
    permute = autograd.permute
    transpose = autograd.transpose
    # arithmetic
    __add__ = __radd__ = autograd.add
    __sub__ = __rsub__ = autograd.sub
    __mul__ = __rmul__ = autograd.mul
    __matmul__ = __rmatmul__ = autograd.matmul
    sum = autograd.sum
