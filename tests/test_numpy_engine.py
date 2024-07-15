""" 
test the numpy backend
"""

from typing import Sequence

import numpy as np
import pytest

from skinnygrad import config, llops, runtime, shapes


def test_numpy_engine_add_reduce() -> None:
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.Ops.LOAD([1, 2, 3])
        y = llops.Ops.LOAD([1, 2, 3])
        z = llops.Ops.MUL(x, y)  # 1, 4, 9
        z_sum = llops.Ops.SUM(z, axes=(0,))  # 14
        res = z_sum.realize().to_python()
    assert res == 14


def test_numpy_engine_assignment() -> None:
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.Ops.LOAD([1, 2, 3])
        x = llops.Ops.ASSIGN(x, llops.Ops.LOAD([4, 5, 6]))
        x_val = x.realize().to_python()
    assert x_val == [4, 5, 6]


def test_numpy_engine_assign_full_loop() -> None:
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.Ops.LOAD([1, 2, 3])
        x = llops.Ops.ASSIGN(x, llops.Ops.ADD(x, llops.Ops.LOAD([1, 1, 1])))
        x = llops.Ops.ASSIGN(x, llops.Ops.ADD(x, llops.Ops.LOAD([1, 1, 1])))
        x_val = x.realize().to_python()
    assert x_val == [3, 4, 5]


@pytest.mark.parametrize(
    "input_data, slice_loc",
    [
        ([[1, 2, 3]], (0, (1, 2))),
        ([[4, 5, 6, 7]], (0, (1, 3))),
        ([[1, 2], [3, 4]], ((0, 2), 1)),
        ([[9, 8, 7], [6, 5, 4], [3, 2, 1]], (1, (1, 3))),
        ([[10, 20], [30, 40]], ((0, 2), (0, 1))),
        ([[10, 20], [30, 40]], (0, 0)),
    ],
)
def test_numpy_engine_slice(
    input_data: llops.PyArrayRepr,
    slice_loc: Sequence[shapes.Loc],
) -> None:
    expected_val = np.array(input_data)[*(i if isinstance(i, int) else slice(*i) for i in slice_loc)].tolist()
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.Ops.LOAD(input_data)
        y = llops.Ops.SELECT(x, loc=slice_loc)
        y_val = y.realize().to_python()
    assert y_val == expected_val


@pytest.mark.parametrize(
    "input_data, pad_width",
    [
        ([[1, 2, 3]], ((1, 0), (2, 2))),  # One row, padding of 1 on both sides and 2 on top and bottom
        ([[4, 5, 6, 7]], ((0, 0), (1, 1))),  # One row, no padding on sides and 1 on top and bottom
        ([[1, 2], [3, 4]], ((1, 1), (1, 0))),  # Two rows, padding of 1 on both sides and 1 on top and bottom
        ([[9, 8, 7], [6, 5, 4], [3, 2, 1]], ((1, 0), (1, 2))),  # Three rows, padding of 1 on top and 2 on bottom
        ([[10, 20], [30, 40]], ((2, 2), (3, 3))),  # Padding of 2 on each side and 3 on top and bottom
        ([[10, 20], [30, 40]], ((0, 0), (0, 0))),  # No padding
    ],
)
def test_numpy_engine_pad(input_data: llops.PyArrayRepr, pad_width: Sequence[tuple[int, int]]) -> None:
    expected_val = np.pad(input_data, pad_width).tolist()
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.Ops.LOAD(input_data)
        y = llops.Ops.PAD(x, *pad_width)
        y_val = y.realize().to_python()
    assert y_val == expected_val
