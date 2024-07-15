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
    # movement
    __getitem__ = autograd.select
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
    # activations
    relu = autograd.relu
    sigmoid = autograd.sigmoid
    softmax = autograd.softmax