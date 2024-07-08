"""

"""

from __future__ import annotations

from typing import ParamSpec

from autodiff import autograd, llops

P = ParamSpec("P")
Gradient = llops.Symbol | None


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


if __name__ == "__main__":
    import numpy as np
    arr1 = [
        [1, 0, 2],
        [-1, 3, 1],
    ]
    arr2 = [
        [3, 1],
        [2, 1],
        [1, 0],
    ]
    
    params1 = Tensor(arr1, requires_grad=True)
    params2 = Tensor(arr2, requires_grad=True)
    result = params1 @ params2
    result.symbol

    assert np.allclose(
        np.array(arr1) @ np.array(arr2),
        result.symbol.realize().to_python(),
    )
