import numpy as np
import pytest

from skinnygrad import config, llops, runtime, tensors


def test_dotprod_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        params = tensors.Tensor([[1, 2, 4], [4, 5, 6], [7, 8, 9]], requires_grad=True)
        a = (params @ [2, 3, 4]).sum()
        a.backprop()
        assert params.gradient is not None
        assert params.gradient.realize().to_python() == [[9, 9, 9]] * 3


def test_matrix_mul_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        params = tensors.Tensor([[1, 2], [3, 4]], requires_grad=True)
        mat = tensors.Tensor([[2, 0], [1, 2]])
        a = (params @ mat).sum()
        a.backprop()
        assert params.gradient is not None
        assert params.gradient.realize().to_python() == [[2, 3], [2, 3]]


def test_non_square_matrix_mul_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        params = tensors.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        mat = tensors.Tensor([[1, 2], [3, 4], [5, 6]])
        a = (params @ mat).sum()
        a.backprop()
        assert (grad := params.gradient) is not None
    assert np.allclose(grad.realize().to_python(), [[3, 7, 11]] * 2)


def test_mul_scalar_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        params = tensors.Tensor([[1, 2], [3, 4]], requires_grad=True)
        scalar = 3
        a = (params * scalar).sum()
        a.backprop()
        assert params.gradient is not None
        assert params.gradient.realize().to_python() == [[scalar, scalar], [scalar, scalar]]


def test_elementwise_multiplication_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        x = tensors.Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = tensors.Tensor([[5, 6], [7, 8]], requires_grad=False)
        z = (x * y).sum()
        z.backprop()
        expected_gradient = [[5, 6], [7, 8]]
        assert x.gradient is not None
        assert np.allclose(x.gradient.realize().to_python(), expected_gradient)


def test_matrix_addition_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        A = tensors.Tensor([[1, 2], [3, 4]], requires_grad=True)
        B = tensors.Tensor([[5, 6], [7, 8]], requires_grad=True)
        C = (A + B).sum()
        C.backprop()
        expected_gradient = [[1, 1], [1, 1]]
        assert A.gradient is not None
        assert B.gradient is not None
        assert np.allclose(A.gradient.realize().to_python(), expected_gradient)
        assert np.allclose(B.gradient.realize().to_python(), expected_gradient)


def test_simple_slice_gradient_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        tensor = tensors.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        sliced_tensor = tensor[0]
        sliced_tensor_sum = sliced_tensor.sum()
        sliced_tensor_sum.backprop()
        expected_gradient = [[1.0, 1.0], [0.0, 0.0]]
        assert tensor.gradient is not None
        assert np.allclose(tensor.gradient.realize().to_python(), expected_gradient)


def test_column_slice_gradient_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        tensor = tensors.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        sliced_tensor = tensor[..., 1]
        sliced_tensor_sum = sliced_tensor.sum()
        sliced_tensor_sum.backprop()
        expected_gradient = [[0.0, 1.0], [0.0, 1.0]]
        assert tensor.gradient is not None
        assert np.allclose(tensor.gradient.realize().to_python(), expected_gradient)


def test_multidim_slice_gradient_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        tensor = tensors.Tensor([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)
        sliced_tensor = tensor[0, (0, 1)]
        sliced_tensor_sum = sliced_tensor.sum()
        sliced_tensor_sum.backprop()
        expected_gradient = [[[1.0], [0.0]], [[0.0], [0.0]]]
        assert tensor.gradient is not None
        assert np.allclose(tensor.gradient.realize().to_python(), expected_gradient)


def test_complex_slice_gradient_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        tensor = tensors.Tensor(np.random.randn(4, 3, 2).tolist(), requires_grad=True)
        sliced_tensor = tensor[(1, 3), ...]
        sliced_tensor_sum = sliced_tensor.sum()
        sliced_tensor_sum.backprop()
        expected_gradient = np.zeros((4, 3, 2))
        expected_gradient[1:3, :] = 1.0
        assert tensor.gradient is not None
        assert np.allclose(tensor.gradient.realize().to_python(), expected_gradient)


def test_single_element_slice_gradient_backprop(engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        tensor = tensors.Tensor([[3.0, 1.0], [4.0, 2.0]], requires_grad=True)
        sliced_tensor = tensor[1, 1]
        sliced_tensor_sum = sliced_tensor
        sliced_tensor_sum.backprop()
        expected_gradient = [[0.0, 0.0], [0.0, 1.0]]
        assert tensor.gradient is not None
        assert np.allclose(tensor.gradient.realize().to_python(), expected_gradient)


@pytest.mark.parametrize(
    "tensor_values, padding, expected_gradient",
    [
        ([[3.0, 1.0], [4.0, 2.0]], [(1, 1), (1, 1)], [[1.0, 1.0], [1.0, 1.0]]),
        ([[3.0, 1.0], [4.0, 2.0]], [(0, 0), (0, 0)], [[1.0, 1.0], [1.0, 1.0]]),
        ([[3.0, 1.0], [4.0, 2.0]], [(2, 0), (1, 3)], [[1.0, 1.0], [1.0, 1.0]]),
        (
            [[[3.0, 1.0], [4.0, 2.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [(1, 1), (1, 1), (1, 1)],
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ),
        ([[3.0, 1.0, 2.0], [4.0, 2.0, 5.0]], [(1, 0), (0, 2)], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        ([[3.0]], [(1, 1), (1, 1)], [[1.0]]),
    ],
)
def test_pad_gradient_backprop(engine: runtime.Engine, tensor_values, padding, expected_gradient) -> None:
    with config.Configuration(engine=engine):
        tensor = tensors.Tensor(tensor_values, requires_grad=True)
        padded_tensor = tensor.pad(padding)
        padded_tensor.sum().backprop()
        assert tensor.gradient is not None
        assert np.allclose(tensor.gradient.realize().to_python(), expected_gradient)


@pytest.mark.parametrize(
    "arr",
    [
        ([1.0]),
        ([2.0]),
        ([0.5]),
        ([[1.0, 2.0], [4.0, 8.0]]),
        ([[[1.0], [2.0]], [[0.5], [0.25]]]),
    ],
)
def test_reciprocal_backward(arr: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr, requires_grad=True)
        t.reciprocal().sum().backprop()
        reciprocal_grad = -1 / (np.array(arr) ** 2)
        assert t.gradient is not None
        assert np.allclose(t.gradient.realize().to_python(), reciprocal_grad.tolist(), atol=1e-6)


@pytest.mark.parametrize(
    "arr",
    [
        ([1], [1]),
        ([0], [1]),
        ([-1], [1]),
        ([[0, 1], [-1, -2]], [[1, 1], [1, 1]]),
        ([[[0.5], [1.5]], [[-0.5], [-1.5]]], [[[1], [1]], [[1], [1]]]),
    ],
)
def test_relu_backward(arr: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr, requires_grad=True)
        t.relu().sum().backprop()

        relu_grad = np.where(np.array(arr) >= 0, 1, 0)
        assert t.gradient is not None
        assert np.allclose(t.gradient.realize().to_python(), relu_grad.tolist(), atol=1e-6)


@pytest.mark.parametrize(
    "arr, grad",
    [
        ([1], [1]),
        ([0], [1]),
        ([-1], [1]),
        ([[0, 1], [-1, -2]], [[1, 1], [1, 1]]),
        ([[[0.5], [1.5]], [[-0.5], [-1.5]]], [[[1], [1]], [[1], [1]]]),
    ],
)
def test_sigmoid_backward(arr: llops.PyArrayRepr, grad: llops.PyArrayRepr, engine: runtime.Engine) -> None:
    with config.Configuration(engine=engine):
        t = tensors.Tensor(arr, requires_grad=True)
        t.sigmoid().sum().backprop()

        sigmoid_values = 1 / (1 + np.exp(-np.array(arr)))
        expected_grad = np.array(grad) * sigmoid_values * (1 - sigmoid_values)
        assert t.gradient is not None
        assert np.allclose(t.gradient.realize().to_python(), expected_grad.tolist(), atol=1e-6)
