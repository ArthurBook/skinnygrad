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
from typing import Callable, Concatenate, Generic, Iterator, ParamSpec, Self, Sequence, TypeVar, Union

from skinnygrad import llops, shapes

# fmt: off
P, T = ParamSpec("P"),TypeVar("T", bound='AutoDiffable')
AutoDiffInput = Union[T, llops.PyArrayRepr]
LazyGrad = Callable[[llops.Symbol], llops.Symbol]
UnaryGradDefinition = Callable[Concatenate[llops.Symbol, P], tuple[llops.Symbol, LazyGrad]]
BinaryGradDefinition = Callable[Concatenate[llops.Symbol, llops.Symbol, P], tuple[llops.Symbol, LazyGrad, LazyGrad]]
UnaryAutoDiffFunc = Callable[Concatenate[AutoDiffInput[T], P], T]
BinaryAutoDiffFunc = Callable[Concatenate[AutoDiffInput[T], AutoDiffInput[T], P], T]
# fmt: on


### Base for autodiff ###
class AutoDiffable(abc.ABC):
    def __init__(self, data: llops.PyArrayRepr | llops.Symbol, requires_grad: bool = False) -> None:
        self.symbol = data if isinstance(data, llops.Symbol) else llops.Ops.LOAD(data)
        self.requires_grad = requires_grad
        self.gradient: llops.Symbol | None = None
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
        self._backward(llops.Ops.BROADCAST(llops.Ops.LOAD(1), shape=self.shape))

    def _backward(self, delta: llops.Symbol) -> None:
        assert self.requires_grad, f"_backward called on tensor with {self.requires_grad=}"
        self.gradient = delta if self.gradient is None else llops.Ops.ADD(self.gradient, delta)
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
def llop_gradient(grad_def: UnaryGradDefinition[P], /) -> UnaryAutoDiffFunc[T, P]: ...
@typing.overload
def llop_gradient(grad_def: BinaryGradDefinition[P], /) -> BinaryAutoDiffFunc[T, P]: ...
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
        common_shape = get_common_broadcast_shape(*iter_autodiffables(sign))
        sign.arguments.update(
            (k, broadcast(ad, common_shape.dims))
            for k, ad in zip(autodiffable_args, iter_autodiffables(sign), strict=True)
            if ad.shape != common_shape
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
def reshape(symbol: llops.Symbol, /, shape: Sequence[int]) -> tuple[llops.Symbol, LazyGrad]:
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
        lost_pads = [  # TODO move this to method on shape
            ((slice_, origin - slice_ - 1) if isinstance(slice_, int) else (slice_[0], origin - slice_[1]))
            for slice_, origin in zip(loc, symbol.shape.dims, strict=True)
        ]
        padded_grad = llops.Ops.PAD(output_grad, *lost_pads)
        assert (padded_grad).shape == symbol.shape, f"{padded_grad.shape=} != {symbol.shape=}"
        return padded_grad

    return forward, backward


@llop_gradient
def pad(symbol: llops.Symbol, pads: Sequence[tuple[int, int]], pad_val: int = 0) -> tuple[llops.Symbol, LazyGrad]:
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


def lpad(autodiffable: AutoDiffInput[T], /, n_dims: int) -> T:
    autodiffable = ensure_autodiffable(autodiffable)
    return addaxes(autodiffable, 0, n_dims)


def rpad(autodiffable: AutoDiffInput[T], /, n_dims: int) -> T:
    autodiffable = ensure_autodiffable(autodiffable)
    return addaxes(autodiffable, autodiffable.shape.ndims, n_dims)


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


@llop_gradient
def neg(symbol: llops.Symbol) -> tuple[llops.Symbol, LazyGrad]:
    forward = llops.Ops.NEG(symbol)
    backward = lambda output_grad: llops.Ops.NEG(output_grad)
    return forward, backward


@llop_gradient
def sigmoid(symbol: llops.Symbol) -> tuple[llops.Symbol, LazyGrad]:
    ones = llops.Ops.BROADCAST(llops.Ops.LOAD(1), shape=symbol.shape)
    forward = llops.Ops.INV(llops.Ops.ADD(ones, llops.Ops.EXP(llops.Ops.NEG(symbol))))

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        sigmoid_grad = llops.Ops.MUL(forward, llops.Ops.ADD(ones, llops.Ops.NEG(forward)))
        return llops.Ops.MUL(output_grad, sigmoid_grad)

    return forward, backward


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
def sum(symbol: llops.Symbol, /, axes: int | Sequence[int] | None = None) -> tuple[llops.Symbol, LazyGrad]:
    axes = (axes,) if isinstance(axes, int) else axes

    def backward(output_grad: llops.Symbol) -> llops.Symbol:
        if axes is not None:
            prev_shape = output_grad.shape.insertaxes(*symbol.shape.normalize_dim_ref(*axes))
            output_grad = llops.Ops.RESHAPE(output_grad, shape=prev_shape)
        output_grad = llops.Ops.BROADCAST(output_grad, shape=symbol.shape)
        assert output_grad.shape == symbol.shape, f"{output_grad.shape=} != {symbol.shape=}"
        return output_grad

    if axes is None:
        flat_symbol = llops.Ops.RESHAPE(symbol, shape=symbol.shape.flat())
        return llops.Ops.SUM(flat_symbol, (0,)), backward
    return llops.Ops.SUM(symbol, tuple(axes)), backward


### helpers ###
def get_common_broadcast_shape(*ads: AutoDiffable) -> shapes.Shape:
    if all(s == ads[0].shape for s in ads):
        return ads[0].shape
    return functools.reduce(lambda s1, s2: s1.broadcast(s2.shape), ads[1:], ads[0].shape)


def ensure_autodiffable(ad: AutoDiffInput[T], /) -> T:
    return ad if isinstance(ad, AutoDiffable) else AutoDiffable(ad)  # type: ignore
