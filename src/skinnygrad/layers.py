import math
from typing import Callable, Final, Iterable, Literal, Protocol, runtime_checkable
from skinnygrad import tensors


@runtime_checkable
class SupportsForward(Protocol):
    def __call__(self, x: tensors.Tensor, /) -> tensors.Tensor: ...


@runtime_checkable
class HasParameters(Protocol):
    def params(self) -> Iterable[tensors.Tensor]: ...


class FFLayer(SupportsForward, HasParameters):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        activation: Callable[[tensors.Tensor], tensors.Tensor] | None = None,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        glorot_ub = math.sqrt(6 / (input_size + output_size))
        self.weights = tensors.Tensor.random_uniform(
            input_size,
            output_size,
            lb=-glorot_ub,
            ub=glorot_ub,
            requires_grad=True,
        )
        self.bias = tensors.Tensor.zeros(output_size, requires_grad=True) if bias else None
        self.activation = activation

    def __call__(self, x: tensors.Tensor) -> tensors.Tensor:
        out = x @ self.weights
        out = out + self.bias if self.bias is not None else out
        return out if self.activation is None else self.activation(out)

    def params(self) -> list[tensors.Tensor]:
        return [self.weights, self.bias] if self.bias is not None else [self.weights]


class ConvLayer(SupportsForward, HasParameters):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        bias: bool = True,
        activation: Callable[[tensors.Tensor], tensors.Tensor] | None = None,
    ) -> None:
        glorot_ub = math.sqrt(6 / (math.prod(kernel_shape) * in_channels + out_channels))
        self.kernel = tensors.Tensor.random_uniform(
            out_channels,
            in_channels,
            *kernel_shape,
            lb=-glorot_ub,
            ub=glorot_ub,
            requires_grad=True,
        )
        self.bias = tensors.Tensor.zeros(out_channels, requires_grad=True) if bias else None
        self.output_channels = out_channels
        self.input_channels = in_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.activation = activation

    def __call__(self, x: tensors.Tensor) -> tensors.Tensor:
        out = x.conv(self.kernel, self.bias, self.stride, self.padding)
        return out if self.activation is None else self.activation(out)

    def params(self) -> list[tensors.Tensor]:
        return [self.kernel, self.bias] if self.bias is not None else [self.kernel]


class PoolingLayer(SupportsForward):
    ReduceMethods = Literal["max", "mean"]
    __reducemethodmap__: Final[dict[ReduceMethods, Callable]] = {
        "max": tensors.Tensor.max,
        "mean": tensors.Tensor.mean,
    }

    def __init__(self, kernel_shape: tuple[int, ...], method: ReduceMethods, stride: int = 1) -> None:
        self.kernel_shape = kernel_shape
        self.reduce_method = self.__reducemethodmap__[method]
        self.reduce_axes = tuple(range(-len(self.kernel_shape), 0))
        self.stride = stride

    def __call__(self, x: tensors.Tensor) -> tensors.Tensor:
        return self.reduce_method(x.pool(self.kernel_shape, self.stride), axes=self.reduce_axes)


### helpers ###
def flatten_except_batch_dim(x: tensors.Tensor) -> tensors.Tensor:
    return x.reshape((x.shape.dims[0], math.prod(x.shape.dims[1:])))
