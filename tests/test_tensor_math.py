import numpy as np
import pytest

from skinnygrad import autograd, config, llops, runtime, tensors


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


@pytest.mark.parametrize(
    "arr1, arr2",
    [
        (1, 2),
        ([[1], [2], [3]], [[[1, 2]]]),
        ([[1, 2, 3]], [[[1]], [[2]], [[3]]]),
        (1, [[[1, 2]]]),
        ([[1, 2]], 1),
        ([[1, 2]], [[1], [2]]),
        ([[0]], [[1]]),  # Zero divided by a number
        ([[1, 2, 3]], [[4, 5, 6]]),  # Element-wise division
        ([[1, 2]], [[3], [4]]),  # Broadcast
        ([[1, 2]], [[[3, 4]]]),  # Three-dimensional with two-dimensional
        ([[-1, -2, -3]], [[1, 2, 3]]),  # Negative numbers
        ([[1.5, 2.5]], [[2, 3]]),  # Floating points
    ],
)
def test_div(arr1: llops.PyArrayRepr, arr2: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    np_result = np.array(arr1) / np.array(arr2)
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr1) / tensors.Tensor(arr2)
        assert np.allclose(t.realize(), np_result.tolist())


@pytest.mark.parametrize(
    "arr1, arr2",
    [
        # Typical square matrices
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1, 0, 2], [-1, 3, 1]], [[3, 1], [2, 1], [1, 0]]),
        # Rectangular matrices
        ([[1, 4, 6], [2, 3, 5]], [[9, 8], [7, 6], [5, 4]]),
        ([[1, 4], [2, 5], [3, 6]], [[1, 2, 3, 4], [5, 6, 7, 8]]),
        # Single row and single column
        ([[1, 2, 3]], [[4], [5], [6]]),
        # Identity matrices
        ([[1, 0], [0, 1]], [[5, 6], [7, 8]]),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        # Matrices with zeros
        ([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
        ([[0, 0, 0], [0, 0, 1], [2, 3, 0]], [[0, 0, 0], [0, 1, 2], [3, 0, 0]]),
        # Matrices with negative numbers
        ([[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]),
        ([[1, -2], [-3, 4]], [[5, -6], [-7, 8]]),
        # Non-square matrices with more rows in arr1
        ([[1, 4, 6], [2, 3, 5], [0, 0, 1]], [[9, 8], [7, 6], [5, 4]]),
        # Non-square matrices with more columns in arr1
        ([[-1, 2, -3], [4, 5, 6]], [[-7, 8, 9], [-6, 5, 4], [3, -2, -1]]),
        # Single-element matrices
        ([[2]], [[3]]),
        ([1], [2]),
    ],
)
def test_matmul_vs_numpy_1(arr1: llops.PyArrayRepr, arr2: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    numpy_result = np.matmul(arr1, arr2)
    with config.Configuration(engine=engine):
        result = autograd.matmul(arr1, arr2).realize()
    assert np.allclose(numpy_result, result), f"{arr1},{arr2}"


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        # Typical square matrices
        ((2, 2), (2, 2)),
        ((2, 3), (3, 2)),
        # Rectangular matrices
        ((2, 3), (3, 2)),
        ((3, 2), (2, 4)),
        # Single row and single column
        ((1, 3), (3, 1)),
        # Identity matrices
        ((2, 2), (2, 2)),
        ((3, 3), (3, 3)),
        # Matrices with zeros
        ((2, 2), (2, 2)),
        ((3, 3), (3, 3)),
        # Large matrices
        ((100, 100), (100, 100)),
        # Matrices with negative numbers
        ((2, 2), (2, 2)),
        ((2, 2), (2, 2)),
        # Non-square matrices with more rows in arr1
        ((3, 3), (3, 2)),
        # Non-square matrices with more columns in arr1
        ((2, 3), (3, 3)),
        # Single-element matrices
        ((1, 1), (1, 1)),
        ((1,), (1,)),
        # 3D Tensors with compatible shapes
        ((2, 3, 4), (2, 4, 3)),
        ((3, 2, 5), (3, 5, 2)),
        ((4, 4, 4), (4, 4, 4)),
        ((3, 4, 5), (3, 5, 4)),
        ((1, 2, 3), (1, 3, 2)),
        # 5D Tensors with compatible shapes
        ((1, 2, 3), (1, 2, 3, 3, 3)),
    ],
)
def test_matmul_vs_numpy_2(shape1: tuple[int, ...], shape2: tuple[int, ...], engine: runtime.Engine) -> None:
    arr1 = np.random.random(shape1)
    arr2 = np.random.random(shape2)
    numpy_result = np.matmul(arr1, arr2)
    with config.Configuration(engine=engine):
        result = autograd.matmul(arr1.tolist(), arr2.tolist()).realize()
    assert np.allclose(numpy_result, result), f"{arr1}, {arr2}"


@pytest.mark.parametrize(
    "arr",
    [
        [1],
        [0],
        [-1],
        [[0, 1], [-1, -2]],
        [[[0.5], [1.5]], [[-0.5], [-1.5]]],
    ],
)
def test_relu_forward(arr: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr).relu()
        expected = np.maximum(0, np.array(arr))
        assert np.allclose(t.symbol.realize().to_python(), expected.tolist(), atol=1e-6)


@pytest.mark.parametrize(
    "arr",
    [
        [1],
        [0],
        [-1],
        [[0, 1], [-1, -2]],
        [[[0.5], [1.5]], [[-0.5], [-1.5]]],
    ],
)
def test_sigmoid_forward(arr: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr).sigmoid()
        expected = 1 / (1 + np.exp(-np.array(arr)))
        assert np.allclose(t.symbol.realize().to_python(), expected.tolist(), atol=1e-6)
