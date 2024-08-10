import abc
from typing import Any, Iterable, Iterator, Self

from skinnygrad import autograd, llops


class Optimizer(abc.ABC):
    def __init__(self, autodiffables: Iterable[autograd.AutoDiffable], *args: Any, **kwargs: Any) -> None:
        super().__init__()

        def filter_params(params: Iterable[autograd.AutoDiffable]) -> Iterator[autograd.AutoDiffable]:
            seen_ids: set[int] = set()
            is_uniq_grad_param = lambda param: param.requires_grad and id(param) not in seen_ids
            for param in filter(is_uniq_grad_param, params):
                seen_ids.add(id(param))
                yield param

        self.params = tuple(filter_params(autodiffables))

    @abc.abstractmethod
    def calc_delta(self) -> Iterable[autograd.AutoDiffable]: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, e_type: type[BaseException] | None, e_value: BaseException | None, e_traceback: Any) -> None:
        if e_type is None:
            self.step()
            self.zero_grad()

    def step(self) -> None:
        for param, delta in zip(self.params, self.calc_delta(), strict=True):
            param.symbol = llops.Ops.ASSIGN(param.symbol, llops.Ops.ADD(param.symbol, delta.symbol))

    def zero_grad(self) -> None:
        for param in self.params:
            param.gradient = None

    def iter_grads(self) -> Iterator[autograd.AutoDiffable]:
        for param in self.params:
            if param.gradient is None:
                raise RuntimeError(
                    f"Gradient for {param} is None."
                    f"Fid you forget to call `{autograd.AutoDiffable.backprop.__qualname__}`?"
                )
            yield param.gradient


class SGD(Optimizer):
    def __init__(self, autodiffables: Iterable[autograd.AutoDiffable], lr: float) -> None:
        super().__init__(autodiffables)
        self.lr = lr

    def calc_delta(self) -> Iterator[autograd.AutoDiffable]:
        return (autograd.mul(-self.lr, grad) for grad in self.iter_grads())
