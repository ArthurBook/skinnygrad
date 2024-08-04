"""
Tensors with torch-like interface
"""

from __future__ import annotations

from skinnygrad import autograd


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


if __name__ == "__main__":
    import numpy as np

    a = Tensor(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
            ],
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
            ],
        ]
    )
    b = a.conv(
        [
            [
                [
                    (1, 2),
                    (3, 4),
                ],
            ],
            [
                [
                    (5, 6),
                    (7, 8),
                ],
            ],
            [
                [
                    (9, 10),
                    (11, 12),
                ],
            ],
        ],
        padding=1,
    )
    np.array(b.realize())
