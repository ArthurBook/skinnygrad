"""
Autodifferentiation logic
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import functools
import inspect
import itertools
import typing
from typing import Callable, Concatenate, Generic, Iterable, Iterator, ParamSpec, Self, Sequence, TypeVar, Union

from skinnygrad import llops, shapes

# fmt: off
P, T, U = ParamSpec("P"), TypeVar("T", bound='AutoDiffable'), TypeVar("U")
AutoDiffInput = Union[T, llops.PyArrayRepr]
LazyGrad = Callable[[llops.Symbol], llops.Symbol]
UnaryGradDef = Callable[Concatenate[llops.Symbol, P], tuple[llops.Symbol, LazyGrad]]
BinaryGradDef = Callable[Concatenate[llops.Symbol, llops.Symbol, P], tuple[llops.Symbol, LazyGrad, LazyGrad]]
TernaryGradDef = Callable[Concatenate[llops.Symbol, llops.Symbol, llops.Symbol, P], tuple[llops.Symbol, LazyGrad, LazyGrad, LazyGrad]]
UnaryAutoDiffFunc = Callable[Concatenate[AutoDiffInput[T], P], T]
BinaryAutoDiffFunc = Callable[Concatenate[AutoDiffInput[T], AutoDiffInput[T], P], T]
TernaryAutoDiffFunc = Callable[Concatenate[AutoDiffInput[T], AutoDiffInput[T], AutoDiffInput[T], P], T]
# fmt: on


### Base for autodiff ###
class AutoDiffable(abc.ABC):
    def __init__(self, data: llops.PyArrayRepr | llops.Symbol, requires_grad: bool = False) -> None:
        self.symbol = data if isinstance(data, llops.Symbol) else llops.Ops.READ(data)
        self.requires_grad = requires_grad
        self.gradient: AutoDiffable | None = None
        self._backprops: tuple[Backprop, ...] = ()

    def __repr__(self) -> str:
        return (
            f"<{self.__module__}.{self.__class__.__name__}(\n"
            f"  {self.symbol!r},\n"
            f"  {self.requires_grad=!r},\n"
            f"  {self.gradient=!r},\n"
            f")>"
        )

    def realize(self) -> llops.PyArrayRepr:
        return self.symbol.realize().to_python()

    def backprop(self) -> None:
        assert self.requires_grad, f"backprop called on tensor with {self.requires_grad=}"
        assert self.shape.size == 1, f"backprop called on tensor with non-scalar {self.shape=}"
        assert self._backprops, f"backprop called on tensor with no grad graph"
        fill_grad = llops.Ops.READ(1)
        if fill_grad.shape != self.shape:
            fill_grad = llops.Ops.BROADCAST(fill_grad, shape=self.shape)
        self._backward(fill_grad)

    def _backward(self, delta: llops.Symbol) -> None:
        assert self.requires_grad, f"_backward called on tensor with {self.requires_grad=}"
        self.gradient = AutoDiffable(delta if self.gradient is None else llops.Ops.ADD(self.gradient.symbol, delta))
        for backprop in self._backprops:
            backprop(delta)

    @property
    def shape(self) -> shapes.Shape:
        return self.symbol.shape

    @classmethod
    def from_backprops(cls, new_symbol: llops.Symbol, backprops: tuple[Backprop[Self], ...]) -> Self:
        self = cls(new_symbol, len(backprops) > 0)
        self._backprops = backprops
        return self


### LLop gradients ###
@dataclasses.dataclass(frozen=True, slots=True)
class Backprop(Generic[T]):
    backward_function: LazyGrad
    backprop_target: T

    def __call__(self, out_grad: llops.Symbol) -> None:
        self.backprop_target._backward(self.backward_function(out_grad))


### llop gradient ⟹ autodiffable function ###
@typing.overload
def llop_gradient(grad_def: UnaryGradDef[P], /) -> UnaryAutoDiffFunc[T, P]: ...
@typing.overload
def llop_gradient(grad_def: BinaryGradDef[P], /) -> BinaryAutoDiffFunc[T, P]: ...
@typing.overload
def llop_gradient(grad_def: TernaryGradDef[P], /) -> TernaryAutoDiffFunc[T, P]: ...
def llop_gradient(grad_def: Callable[P, tuple[llops.Symbol, LazyGrad]]) -> Callable[P, AutoDiffable]:  # type: ignore
    """
    Turn the symbol-level def of forward & backward into an autodiffable function for tensors
    """

    ## parse the signature
    sign = inspect.signature(grad_def)
    types = typing.get_type_hints(grad_def)
    autodiffable_args = tuple(k for k in sign.parameters if types[k] is llops.Symbol)

    @functools.wraps(grad_def)
    def autodiffable_function(*args: P.args, **kwargs: P.kwargs) -> AutoDiffable:
        (bound_args := sign.bind(*args, **kwargs)).apply_defaults()
        validate_autodiffable_args(bound_args)
        broadcast_autodiffable_args(bound_args)
        return create_output_autodiffable(bound_args)

    def validate_autodiffable_args(sign: inspect.BoundArguments) -> None:
        sign.arguments.update(zip(autodiffable_args, iter_autodiffables(sign), strict=True))

    def broadcast_autodiffable_args(sign: inspect.BoundArguments) -> None:
        shape = get_common_broadcast_shape(*iter_autodiffables(sign))
        sign.arguments.update(
            (k, broadcast(ad, shape.dims))
            for k, ad in zip(autodiffable_args, iter_autodiffables(sign), strict=True)
            if ad.shape != shape
        )

    def create_output_autodiffable(sign: inspect.BoundArguments) -> AutoDiffable:
        args_symbol = create_copy_with_symbols(sign)
        new_symbol, *grads = grad_def(*args_symbol.args, **args_symbol.kwargs)
        grads_with_targets = zip(grads, iter_autodiffables(sign), strict=True)
        backprops = (Backprop(f, tgt) for f, tgt in grads_with_targets if tgt.requires_grad)
        return next(iter_autodiffables(sign)).from_backprops(new_symbol, tuple(backprops))

    def create_copy_with_symbols(sign: inspect.BoundArguments) -> inspect.BoundArguments:
        sign_with_symbols = inspect.BoundArguments(sign.signature, arg_copy := copy.copy(sign.arguments))
        arg_copy.update((k, v.symbol) for k, v in zip(autodiffable_args, iter_autodiffables(sign), strict=True))
        return sign_with_symbols

    def iter_autodiffables(bound_args: inspect.BoundArguments) -> Iterator[AutoDiffable]:
        return (ensure_autodiffable(bound_args.arguments[k]) for k in autodiffable_args)

    return autodiffable_function


### Unary gradient defs ###
@llop_gradient
def reshape(symbol: llops.Symbol, /, shape: Iterable[int]) -> tuple[llops.Symbol, LazyGrad]:
    if symbol.shape == (newshape := shapes.Shape(tuple(shape))):
        return symbol, lambda output_grad: output_grad  # no op needed
    forward = llops.Ops.RESHAPE(symbol, shape=newshape)
    backward = lambda output_grad: llops.Ops.RESHAPE(output_grad, shape=symbol.shape)
    return forward, backward


@llop_gradient
def select(symbol: llops.Symbol, loc: shapes.Loc | tuple[shapes.Loc, ...]) -> tuple[llops.Symbol, LazyGrad]:
    loc = symbol.shape.normalize_loc(loc if isinstance(loc, tuple) else (loc,))
    forward = llops.Ops.SELECT(symbol, loc=loc, _skip_norm=True)

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        if lost_dims := tuple(i for i, d in enumerate(loc) if isinstance(d, int)):
            output_grad = llops.Ops.RESHAPE(output_grad, shape=output_grad.shape.insertaxes(*lost_dims))
        lost_pads = [
            ((slice_, origin - slice_ - 1) if isinstance(slice_, int) else (slice_[0], origin - slice_[1]))
            for slice_, origin in zip(loc, symbol.shape.dims, strict=True)
        ]
        padded_grad = llops.Ops.PAD(output_grad, *lost_pads)
        assert (padded_grad).shape == symbol.shape, f"{padded_grad.shape=} != {symbol.shape=}"
        return padded_grad

    return forward, backward


@llop_gradient
def pad(
    symbol: llops.Symbol,
    pads: Sequence[int | tuple[int, int]],
    pad_val: float = 0,
) -> tuple[llops.Symbol, LazyGrad]:
    pads = [(p, p) if isinstance(p, int) else p for p in pads]

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        slices = [(lpad, lpad + original) for (lpad, _), original in zip(pads, symbol.shape.dims, strict=True)]
        grad_slice = llops.Ops.SELECT(output_grad, loc=slices, _skip_norm=True)
        assert grad_slice.shape == symbol.shape, f"{grad_slice.shape=} != {symbol.shape=}"
        return grad_slice

    forward = llops.Ops.PAD(symbol, *pads, pad_val=pad_val)
    return forward, backward


def addaxes(autodiffable: AutoDiffInput[T], /, idx: int, n_dims: int) -> T:
    autodiffable = ensure_autodiffable(autodiffable)
    return reshape(autodiffable, autodiffable.shape.addaxes(idx, n_dims).dims)


def flatten(autodiffable: AutoDiffInput[T], /) -> T:
    autodiffable = ensure_autodiffable(autodiffable)
    return reshape(autodiffable, autodiffable.shape.flat().dims)


@llop_gradient
def permute(symbol: llops.Symbol, /, *order: int) -> tuple[llops.Symbol, LazyGrad]:
    """
    Transpose tensor. for order=None, order:=symbol.shape[::-1]
    a la [numpy](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
    """

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        reversed_order = tuple(sorted(range(len(order)), key=order.__getitem__))
        output_grad = llops.Ops.PERMUTE(output_grad, order=reversed_order)
        assert output_grad.shape == symbol.shape, f"{output_grad.shape=} != {symbol.shape=}"
        return output_grad

    order = order if order else tuple(reversed(range(symbol.shape.ndims)))
    forward = llops.Ops.PERMUTE(symbol, order=order)
    return forward, backward


def transpose(autodiffable: AutoDiffInput[T], /, dim1: int, dim2: int) -> T:
    autodiffable = ensure_autodiffable(autodiffable)
    axes = list(range(autodiffable.shape.ndims))
    axes[dim1], axes[dim2] = axes[dim2], axes[dim1]
    return permute(autodiffable, *axes)


@llop_gradient
def broadcast(symbol: llops.Symbol, /, shape: Sequence[int]) -> tuple[llops.Symbol, LazyGrad]:
    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        dims = itertools.zip_longest(reversed(shape), reversed(symbol.shape.dims), fillvalue=0)
        new_dims = tuple(-1 - idx for idx, (i, j) in enumerate(dims) if i != j)
        output_grad = llops.Ops.SUM(output_grad, new_dims)
        if symbol.shape != output_grad.shape:
            output_grad = llops.Ops.RESHAPE(output_grad, shape=symbol.shape)
        return output_grad

    forward = llops.Ops.BROADCAST(symbol, shape=shapes.Shape(tuple(shape)))
    return forward, backward


def repeat(autodiffable: AutoDiffInput[T], repeats: tuple[int, ...]) -> T:
    autodiffable = ensure_autodiffable(autodiffable)
    base = (1,) * (len(repeats) - autodiffable.shape.ndims) + autodiffable.shape.dims
    new_shape = tuple(itertools.chain.from_iterable([1, b] for b in base))
    expand_shape = tuple(itertools.chain.from_iterable(zip(repeats, base)))
    final_shape = tuple(r * s for r, s in zip(repeats, base))
    return reshape(broadcast(reshape(autodiffable, new_shape), expand_shape), final_shape)


def pool(
    autodiffable: AutoDiffInput[T],
    /,
    kernel_shape: Sequence[int],
    stride: int | tuple[int, ...] = 1,
) -> T:
    autodiffable = ensure_autodiffable(autodiffable)
    stride = len(kernel_shape) * (stride,) if isinstance(stride, int) else stride
    assert all(i > 0 for i in kernel_shape), f"{kernel_shape=} must be positive"
    assert all(i > 0 for i in stride), f"{stride=} must be positive"
    assert (n_kernel_dims := len(kernel_shape)) == len(stride), f"{kernel_shape=} mismatch {stride=}"
    assert n_kernel_dims <= autodiffable.shape.ndims, f"{autodiffable.shape=} mismatch {kernel_shape=}"
    orig_noop_shape, orig_dims = autodiffable.shape.dims[:-n_kernel_dims], autodiffable.shape.dims[-n_kernel_dims:]
    noop_pad = (1,) * (n_noop_dims := autodiffable.shape.ndims - n_kernel_dims)
    mvmnt = tuple(1 + (orig - kern) // strd for orig, kern, strd in zip(orig_dims, kernel_shape, stride))
    pool = repeat(autodiffable, noop_pad + tuple(k + 1 for k, o in zip(kernel_shape, orig_dims)))
    pool = select(pool, (..., *[(0, k * (o + 1)) for o, k in zip(orig_dims, kernel_shape)]))
    pool = reshape(pool, orig_noop_shape + _flatten((k, o + 1) for o, k in zip(orig_dims, kernel_shape)))
    pool = select(pool, (..., *_flatten(((0, k), (0, m * s)) for k, m, s in zip(kernel_shape, mvmnt, stride))))
    pool = reshape(pool, orig_noop_shape + _flatten(zip(kernel_shape, mvmnt, stride)))
    pool = select(pool, (..., *_flatten(((0, k), (0, m), (0, 1)) for k, m in zip(kernel_shape, mvmnt))))
    pool = reshape(pool, orig_noop_shape + _flatten(zip(kernel_shape, mvmnt)))
    return permute(
        pool,
        *range(n_noop_dims),
        *(n_noop_dims + 2 * i + 1 for i in range(len(orig_dims))),
        *(n_noop_dims + 2 * i for i in range(len(orig_dims))),
    )


def conv(
    autodiffable: AutoDiffInput[T],  # (batch_size, in_channels, *input_shape)
    kernel: AutoDiffInput[T],  # (out_channels, in_channels, *kernel_shape)
    bias: AutoDiffInput[T] | None = None,  # (out_channels, )
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    pad_val: float = 0,
) -> T:  # (batch_size, out_channels, *pool(input_shape+padding, kernel_shape, stride))
    """ """
    autodiffable, kernel = ensure_autodiffable(autodiffable), ensure_autodiffable(kernel)
    batch_size, input_in_channels, *input_shape = autodiffable.shape.dims
    out_channels, kern_in_channels, *conv_shape = kernel.shape.dims
    assert len(input_shape) == len(conv_shape), f"Mismatch in shapes along conv {input_shape=}, {conv_shape=}"
    assert input_in_channels == kern_in_channels, f"Mismatch input channels {input_in_channels=} {kern_in_channels=}"
    if any(padding := ((padding,) if isinstance(padding, int) else padding)):
        autodiffable = pad(autodiffable, (0,) * (autodiffable.shape.ndims - len(padding)) + padding, pad_val)
    pools = pool(autodiffable, kernel_shape=conv_shape, stride=stride)
    bc_pools = reshape(pools, (batch_size, 1, *pools.shape.dims[1:]))
    bc_kernel = reshape(kernel, (1, out_channels, kern_in_channels, *(1,) * len(conv_shape), *conv_shape))
    out = sum(mul(bc_pools, bc_kernel), axes=(2,) + tuple(range(-len(conv_shape), 0)))
    assert out.shape.dims == (batch_size, out_channels, *pools.shape.dims[2 : 2 + len(conv_shape)]), "Shape error"
    return out if bias is None else add(out, reshape(bias, (1, out_channels, *(1 for _ in conv_shape))))


@llop_gradient
def neg(symbol: llops.Symbol) -> tuple[llops.Symbol, LazyGrad]:
    forward = llops.Ops.NEG(symbol)
    backward = lambda output_grad: llops.Ops.NEG(output_grad)
    return forward, backward


@llop_gradient
def exp(symbol: llops.Symbol) -> tuple[llops.Symbol, LazyGrad]:
    forward = llops.Ops.EXP(symbol)
    backward = lambda output_grad: llops.Ops.MUL(forward, output_grad)
    return forward, backward


@llop_gradient
def reciprocal(symbol: llops.Symbol) -> tuple[llops.Symbol, LazyGrad]:
    forward = llops.Ops.INV(symbol)

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        reciprocal_grad = llops.Ops.NEG(llops.Ops.INV(llops.Ops.MUL(symbol, symbol)))
        return llops.Ops.MUL(reciprocal_grad, output_grad)

    return forward, backward


@llop_gradient
def relu(symbol1: llops.Symbol) -> tuple[llops.Symbol, LazyGrad]:
    zeros = llops.Ops.BROADCAST(llops.Ops.READ(0), shape=symbol1.shape)
    forward = llops.Ops.WHERE(cond := llops.Ops.LESS(symbol1, zeros), zeros, symbol1)
    backward = lambda output_grad: llops.Ops.WHERE(cond, zeros, output_grad)
    return forward, backward


@llop_gradient
def sigmoid(symbol: llops.Symbol) -> tuple[llops.Symbol, LazyGrad]:
    ones = llops.Ops.BROADCAST(llops.Ops.READ(1), shape=symbol.shape)
    forward = llops.Ops.INV(llops.Ops.ADD(ones, llops.Ops.EXP(llops.Ops.NEG(symbol))))

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        sigmoid_grad = llops.Ops.MUL(forward, llops.Ops.ADD(ones, llops.Ops.NEG(forward)))
        return llops.Ops.MUL(output_grad, sigmoid_grad)

    return forward, backward


def softmax(ad: AutoDiffInput[T], /, axes: int | Sequence[int] | None = None) -> T:
    ad = ensure_autodiffable(ad)
    axes = parse_axes(axes, ad.shape)
    t_exp = exp(sub(ad, amax(ad, axes, keepdims=True)))
    t_sum = sum(t_exp, axes, keepdims=True)
    return div(t_exp, t_sum)


### Binary gradient defs ###
@llop_gradient
def add(symbol1: llops.Symbol, symbol2: llops.Symbol) -> tuple[llops.Symbol, LazyGrad, LazyGrad]:
    forward = llops.Ops.ADD(symbol1, symbol2)
    backward = lambda output_grad: output_grad
    return forward, backward, backward


def sub(ad1: AutoDiffInput[T], ad2: AutoDiffInput[T], /) -> T:
    return add(ad1, neg(ad2))


@llop_gradient
def mul(symbol1: llops.Symbol, symbol2: llops.Symbol) -> tuple[llops.Symbol, LazyGrad, LazyGrad]:
    forward = llops.Ops.MUL(symbol1, symbol2)
    backward1 = lambda output_grad: llops.Ops.MUL(output_grad, symbol2)
    backward2 = lambda output_grad: llops.Ops.MUL(output_grad, symbol1)
    return forward, backward1, backward2


def div(ad1: AutoDiffInput[T], ad2: AutoDiffInput[T], /) -> T:
    return mul(ad1, reciprocal(ad2))


@llop_gradient
def pow(symbol1: llops.Symbol, symbol2: llops.Symbol) -> tuple[llops.Symbol, LazyGrad, LazyGrad]:
    forward = llops.Ops.POW(symbol1, symbol2)

    def backward1(output_grad: llops.Symbol) -> llops.Symbol:
        new_exponent = llops.Ops.ADD(symbol2, llops.Ops.BROADCAST(llops.Ops.READ(-1), shape=symbol2.shape))
        return llops.Ops.MUL(output_grad, llops.Ops.MUL(symbol2, llops.Ops.POW(symbol1, new_exponent)))

    def backward2(output_grad: llops.Symbol) -> llops.Symbol:
        return llops.Ops.MUL(output_grad, llops.Ops.MUL(forward, llops.Ops.LOG(symbol1)))

    return forward, backward1, backward2


def matmul(ad1: AutoDiffInput[T], ad2: AutoDiffInput[T], /) -> T:
    """
    Matrix product of ad1 and ad2.
    Same broadcasting logic as [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
    """
    t1, t2 = ensure_autodiffable(ad1), ensure_autodiffable(ad2)
    assert t1.shape.ndims > 0 and t2.shape.ndims > 0, f"matmul requires {t1=} and {t2=} to have at least 1 dim"
    t1_bc, t2_bc = addaxes(t1, -1, 1), transpose(addaxes(t2, -2, 1), -1, -2)
    return sum(mul(t1_bc, t2_bc), -1)


### reduce gradient defs ###
@llop_gradient
def sum(
    symbol: llops.Symbol, /, axes: int | Sequence[int] | None = None, keepdims: bool = False
) -> tuple[llops.Symbol, LazyGrad]:
    axes = parse_axes(axes, symbol.shape)
    forward = llops.Ops.SUM(symbol, axes=axes)
    if keepdims:
        forward = llops.Ops.RESHAPE(forward, shape=forward.shape.insertaxes(*axes))

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        if not keepdims and len(axes) < len(symbol.shape):  # NOTE: add back dropped dims before expand
            output_grad = llops.Ops.RESHAPE(output_grad, shape=output_grad.shape.insertaxes(*axes))
        return llops.Ops.BROADCAST(output_grad, shape=symbol.shape)

    return forward, backward


@llop_gradient
def amax(
    symbol: llops.Symbol, /, axes: int | Sequence[int] | None = None, keepdims: bool = False
) -> tuple[llops.Symbol, LazyGrad]:
    axes = parse_axes(axes, symbol.shape)
    forward = llops.Ops.AMAX(symbol, axes=axes)
    if keepdims:
        forward = llops.Ops.RESHAPE(forward, shape=forward.shape.insertaxes(*axes))

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        nonlocal forward
        if not keepdims and len(axes) < len(symbol.shape):  # NOTE: add back dropped dims before expand
            forward = llops.Ops.RESHAPE(forward, shape=forward.shape.insertaxes(*axes))
            output_grad = llops.Ops.RESHAPE(output_grad, shape=output_grad.shape.insertaxes(*axes))
        return llops.Ops.WHERE(
            llops.Ops.EQ(symbol, llops.Ops.BROADCAST(forward, shape=symbol.shape)),
            llops.Ops.BROADCAST(output_grad, shape=symbol.shape),
            llops.Ops.BROADCAST(llops.Ops.READ(0), shape=symbol.shape),
        )

    return forward, backward


### Ternary gradient defs ###
@llop_gradient
def where(s1: llops.Symbol, s2: llops.Symbol, s3: llops.Symbol) -> tuple[llops.Symbol, LazyGrad, LazyGrad, LazyGrad]:
    def backward1(_: llops.Symbol) -> llops.Symbol:
        raise NotImplementedError("No gradient defined for where condition")

    def backward2(output_grad: llops.Symbol) -> llops.Symbol:
        return llops.Ops.WHERE(s1, output_grad, llops.Ops.BROADCAST(llops.Ops.READ(0), shape=output_grad.shape))

    def backward3(output_grad: llops.Symbol) -> llops.Symbol:
        return llops.Ops.WHERE(s1, llops.Ops.BROADCAST(llops.Ops.READ(0), shape=output_grad.shape), output_grad)

    return llops.Ops.WHERE(s1, s2, s3), backward1, backward2, backward3


### helpers ###
def get_common_broadcast_shape(*ads: AutoDiffable) -> shapes.Shape:
    if all(s == ads[0].shape for s in ads):
        return ads[0].shape
    return functools.reduce(lambda s1, s2: s1.broadcast(s2.shape), ads[1:], ads[0].shape)


def parse_axes(axes: int | Sequence[int] | None, shape: shapes.Shape) -> tuple[int, ...]:
    match axes:
        case int(axis):
            return shape.normalize_dim_ref(axis)
        case tuple(axes):
            return shape.normalize_dim_ref(*axes)
        case None:
            return tuple(range(shape.ndims))
        case _:
            raise TypeError(f"Expected int, Sequence[int], or None, got {axes=}")


def ensure_autodiffable(ad: AutoDiffInput[T], /) -> T:
    return ad if isinstance(ad, AutoDiffable) else AutoDiffable(ad)  # type: ignore


def _flatten(x_: Iterable[Iterable[U]]) -> tuple[U, ...]:
    return tuple(itertools.chain.from_iterable(x_))
