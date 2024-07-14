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
    # arithmetic
    __add__ = __radd__ = autograd.add
    __sub__ = __rsub__ = autograd.sub
    __mul__ = __rmul__ = autograd.mul
    __matmul__ = __rmatmul__ = autograd.matmul
    sum = autograd.sum


if __name__ == "__main__":
    import numpy as np

    a = Tensor([[[0], [0], [0]]], requires_grad=True)
    b = (a[0, (1, 3)] * [[1], [2]]).sum()
    b.backprop()
    np.allclose(a.gradient.realize().to_python(), [[[0], [1], [2]]])
