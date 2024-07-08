import numpy as np
import pytest

from autodiff import config, llops, runtime, tensors


@pytest.mark.parametrize(
    "arr1, arr2",
    [
        (1, 2),
        ([[1], [2], [3]], [[[1, 2]]]),
        ([[1, 2, 3]], [[[1]], [[2]], [[3]]]),
        (1, [[[1, 2]]]),
        ([[1, 2]], 1),
        ([[1, 2]], [[1], [2]]),
    ],
)
def test_add(arr1: llops.PyArrayRepr, arr2: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr1) + tensors.Tensor(arr2)
        assert t.symbol.realize().to_python() == (np.array(arr1) + np.array(arr2)).tolist()


@pytest.mark.parametrize(
    "arr1, arr2",
    [
        (1, 2),
        ([[1], [2], [3]], [[[1, 2]]]),
        ([[1, 2, 3]], [[[1]], [[2]], [[3]]]),
        (1, [[[1, 2]]]),
        ([[1, 2]], 1),
        ([[1, 2]], [[1], [2]]),
    ],
)
def test_sub(arr1: llops.PyArrayRepr, arr2: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr1) - tensors.Tensor(arr2)
        assert t.symbol.realize().to_python() == (np.array(arr1) - np.array(arr2)).tolist()


@pytest.mark.parametrize(
    "arr1, arr2",
    [
        (1, 2),
        ([[1], [2], [3]], [[[1, 2]]]),
        ([[1, 2, 3]], [[[1]], [[2]], [[3]]]),
        (1, [[[1, 2]]]),
        ([[1, 2]], 1),
        ([[1, 2]], [[1], [2]]),
    ],
)
def test_mul(arr1: llops.PyArrayRepr, arr2: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr1) * tensors.Tensor(arr2)
        assert t.symbol.realize().to_python() == (np.array(arr1) * np.array(arr2)).tolist()
