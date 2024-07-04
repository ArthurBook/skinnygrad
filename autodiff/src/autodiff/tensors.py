"""
"""

from __future__ import annotations

import copy
import functools
from typing import Generic, Iterable, ParamSpec, Protocol, Sequence

from autodiff import llops

P = ParamSpec("P")
Gradient = llops.Symbol | None


###
# Tensor
###
class Tensor:
    __slot__ = "symbol", "requires_grad", "src_function", "gradient"

    def __init__(
        self,
        data: llops.Symbol | llops.PyArrayRepr,
        requires_grad: bool = False,
        src_function: AutoDiffFunction | None = None,
    ) -> None:
        self.symbol = data if isinstance(data, llops.Symbol) else llops.ControlOps.LOAD(data)
        self.requires_grad = requires_grad
        self.src_function = src_function
        self.gradient: Tensor | None = None

    def __repr__(self) -> str:
        return (
            f"Tensor(\n"
            f"  {self.symbol},\n"
            f"  requires_grad={'yes' if self.requires_grad else 'no'},\n"
            f"  gradient={self.gradient!r},\n"
            f"  src_function={self.src_function!r}\n"
            f")"
        )

    def __neg__(self) -> Tensor:
        return Negative((self,))

    def __add__(self, other: Tensor | llops.PyArrayRepr, /) -> Tensor:
        return Add((self, other if isinstance(other, Tensor) else Tensor(other)))

    def __radd__(self, other: Tensor | llops.PyArrayRepr, /) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | llops.PyArrayRepr, /) -> Tensor:
        return self + -(other if isinstance(other, Tensor) else Tensor(other))

    def __rsub__(self, other: Tensor | llops.PyArrayRepr, /) -> Tensor:
        return self - other

    def __mul__(self, other: Tensor | llops.PyArrayRepr, /) -> Tensor:
        return Multiply((self, (other if isinstance(other, Tensor) else Tensor(other))))

    def __rmul__(self, other: Tensor | llops.PyArrayRepr, /) -> Tensor:
        return self * other

    def __matmul__(self, other: Tensor | llops.PyArrayRepr, /) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        assert (n1 := self.ndims) > 0 and (n2 := other.ndims) > 0, "Cant do matmul on scalars"
        assert (self.shape[-1]) == (other.shape[-min(n2, 2)]), f"{self.shape=}<->{other.shape=}"
        x = self.reshape((*self.shape[0:-1], *[1] * min(n1 - 1, n2 - 1, 1), self.shape[-1]))
        other = other.reshape(
            (*other.shape[0:-2], *[1] * min(n1 - 1, n2 - 1, 1), *other.shape[-min(n2, 2) :])
        ).transpose(-1, -min(n2, 2))
        return (x * other).sum(-1)

    def sum(self, axes: int | Sequence[int] | None = None) -> Tensor:
        axes = (axes,) if isinstance(axes, int) else axes
        return Sum((self,), axes=axes)

    ### shapes ###
    def reshape(self, shape: Iterable[int]) -> Tensor:
        return Reshape((self,), shape=tuple(shape))

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        order = list(range(len(self.shape)))
        order[dim1], order[dim2] = order[dim2], order[dim1]
        return Permute((self,), order=order)

    @property
    def T(self) -> Tensor:
        return self.transpose(1, 0)

    ### backwards
    def backward(self) -> None:
        assert self.shape.size == 1, "backward() called on tensor with more than one element"
        assert self.requires_grad, "backward() called on tensor that does not require grad"
        assert self.src_function is not None, "backward() called on tensor that has no src_function"
        self.src_function.backward(Tensor(1))

    @property
    def shape(self) -> llops.Shape:
        return self.symbol.shape

    @property
    def ndims(self) -> int:
        return self.shape.ndims


###
# Functions
###
class AutoDiffFunction(Generic[P]):
    __slots__ = "cls", "function", "backprop_tensors", "output_tensor"
    cls: type[SupportsAutoDiff[P]]  # decorated class
    ## after __call__:
    function: SupportsAutoDiff[P]
    backprop_tensors: tuple[Tensor | None, ...]

    def __init__(self, cls: type[SupportsAutoDiff[P]]) -> None:
        self.cls = cls

    def __call__(self, inputs: tuple[Tensor, ...], *args: P.args, **kwargs: P.kwargs) -> Tensor:
        new_self = copy.copy(self)
        new_self.function = new_self.cls(*args, **kwargs)
        new_self.backprop_tensors = tuple(t if t.requires_grad else None for t in inputs)
        return Tensor(
            data=new_self.function.forward(*[s.symbol for s in inputs]),
            requires_grad=any(s.requires_grad for s in inputs),
            src_function=new_self,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.cls.__name__})"

    def __repr__(self) -> str:
        return str(self)

    def backward(self, out_grad: Tensor) -> None:
        reqs_grad = (s is not None for s in self.backprop_tensors)
        backprop_grads = self.function.backward(out_grad.symbol, *reqs_grad)
        for tensor, grad_symbol in zip(self.backprop_tensors, backprop_grads):
            if grad_symbol is not None and tensor is not None:
                grad = Tensor(grad_symbol, requires_grad=False)
                tensor.gradient = grad if tensor.gradient is None else tensor.gradient + grad
                if tensor.src_function is not None:
                    tensor.src_function.backward(grad)


### Implementations ###
class SupportsAutoDiff(Protocol[P]):
    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...
    def forward(self, *symbols: llops.Symbol) -> llops.Symbol: ...
    def backward(self, grad: llops.Symbol, *req_grad: bool) -> tuple[Gradient, ...]: ...


### View ###
@AutoDiffFunction
class Reshape:
    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = llops.Shape(tuple(shape))

    def forward(self, src: llops.Symbol, *_) -> llops.Symbol:
        self.input_shape = src.shape
        return llops.ControlOps.RESHAPE(src, self.shape)

    def backward(self, grad: llops.Symbol, *reqs_grad: bool) -> tuple[Gradient]:
        return (llops.ControlOps.RESHAPE(grad, self.input_shape) if reqs_grad[0] else None,)


@AutoDiffFunction
class Broadcast:
    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = llops.Shape(tuple(shape))
        self.new_ndims = len(self.shape)

    def forward(self, src: llops.Symbol, *_) -> llops.Symbol:
        self.expanded_dim_idxs = list[int]()
        for i, (src_dim, dst_dim) in enumerate(zip(src.shape[::-1], self.shape[::-1])):
            if src_dim != dst_dim:
                assert src_dim == 1, f"Cannot broadcast {src.shape=} to {self.shape=}"
                self.expanded_dim_idxs.append(self.new_ndims - i - 1)
        return llops.ControlOps.EXPAND(src, self.shape)

    def backward(self, grad: llops.Symbol, reqs_grad: bool, *_) -> tuple[Gradient]:
        if not reqs_grad:
            return (None,)
        return (llops.ReduceOps.SUM(grad, tuple(self.expanded_dim_idxs)),)


@AutoDiffFunction
class Permute:
    def __init__(self, order: Sequence[int]) -> None:
        self.order = order
        self.reversed_order = sorted(range(len(order)), key=order.__getitem__)

    def forward(self, src: llops.Symbol, *_) -> llops.Symbol:
        assert len(self.order) == len(src.shape), f"{self.order=} must have same len as {src.shape}"
        return llops.ControlOps.PERMUTE(src, self.order)

    def backward(self, grad: llops.Symbol, *reqs_grad: bool) -> tuple[Gradient]:
        return (llops.ControlOps.PERMUTE(grad, self.reversed_order) if reqs_grad[0] else None,)


@AutoDiffFunction
class Flatten:
    def forward(self, src: llops.Symbol, *_) -> llops.Symbol:
        self.input_shape = src.shape
        return llops.ControlOps.RESHAPE(src, src.shape.flat())

    def backward(self, grad: llops.Symbol, *reqs_grad: bool) -> tuple[Gradient]:
        return (llops.ControlOps.RESHAPE(grad, self.input_shape) if reqs_grad[0] else None,)


### Unary ###
@AutoDiffFunction
class Negative:
    @staticmethod
    def forward(src: llops.Symbol, *_) -> llops.Symbol:
        return llops.UnaryOps.NEG(src)

    @staticmethod
    def backward(grad: llops.Symbol, reqs_grad: bool, *_) -> tuple[Gradient]:
        return (llops.UnaryOps.NEG(grad) if reqs_grad else None,)


### Binary ###
class BroadcastMixin:
    def broadcast_forward(self, *srcs: llops.Symbol) -> tuple[llops.Symbol, ...]:
        self.broadcast_ops = list[SupportsAutoDiff | None]()
        broadcasted_srcs = list[llops.Symbol]()
        for src in srcs:
            if src.shape != (broadcast_shape := get_broadcast_shape(*srcs)):
                self.broadcast_ops.append(broadcast := Broadcast.cls(shape=broadcast_shape))
                broadcasted_srcs.append(broadcast.forward(src))
            else:
                self.broadcast_ops.append(None)
                broadcasted_srcs.append(src)
        return tuple(broadcasted_srcs)

    def broadcast_backward(self, grad: llops.Symbol, src_idx: int) -> llops.Symbol:
        broadcast = self.broadcast_ops[src_idx]
        broadcasted_grad = grad if broadcast is None else broadcast.backward(grad, True)[0]
        assert broadcasted_grad is not None, "Broadcasted grad is None"  # for mypy
        return broadcasted_grad


@AutoDiffFunction
class Add(BroadcastMixin):
    def forward(self, src1: llops.Symbol, src2: llops.Symbol, *_) -> llops.Symbol:
        self.src1, self.src2 = self.broadcast_forward(src1, src2)
        return llops.BinaryOps.ADD(self.src1, self.src2)

    def backward(
        self, grad: llops.Symbol, req_1: bool, req_2: bool, *_
    ) -> tuple[Gradient, Gradient]:
        return (
            (self.broadcast_backward(grad, 0) if req_1 else None),
            (self.broadcast_backward(grad, 1) if req_2 else None),
        )


@AutoDiffFunction
class Multiply(BroadcastMixin):
    def forward(self, src1: llops.Symbol, src2: llops.Symbol, *_) -> llops.Symbol:
        self.src1, self.src2 = self.broadcast_forward(src1, src2)
        return llops.BinaryOps.MUL(self.src1, self.src2)

    def backward(
        self, grad: llops.Symbol, req_1: bool, req_2: bool, *_
    ) -> tuple[Gradient, Gradient]:
        return (
            (self.broadcast_backward(llops.BinaryOps.MUL(grad, self.src2), 0) if req_1 else None),
            (self.broadcast_backward(llops.BinaryOps.MUL(grad, self.src1), 1) if req_2 else None),
        )


### Reduce ###
@AutoDiffFunction
class Sum:
    def __init__(self, axes: Sequence[int] | None = None) -> None:
        self.axis = axes

    def forward(self, src: llops.Symbol, *_) -> llops.Symbol:
        self.input_shape = src.shape
        if self.axis is None:
            return llops.ReduceOps.SUM(llops.ControlOps.RESHAPE(src, src.shape.flat()), (0,))
        return llops.ReduceOps.SUM(src, self.axis)

    def backward(self, grad: llops.Symbol, reqs_grad: bool, *_) -> tuple[Gradient]:
        if self.axis is not None and reqs_grad:
            dropped_axes = self.input_shape.normalize_idxs(*self.axis)
            grad = llops.ControlOps.RESHAPE(grad, grad.shape.addaxes(dropped_axes))
        return (llops.ControlOps.EXPAND(grad, self.input_shape) if reqs_grad else None,)


###
# Helpers
###
def get_broadcast_shape(*src: llops.Symbol) -> llops.Shape:
    if all(s.shape == src[0].shape for s in src):
        return src[0].shape
    return functools.reduce(lambda s, symbol: s.broadcast(symbol.shape), src[1:], src[0].shape)


if __name__ == "__main__":
    ...
