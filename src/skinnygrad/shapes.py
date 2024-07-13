"""
Shape tracking
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Iterator, Sequence

if TYPE_CHECKING:
    from skinnygrad import llops


@dataclasses.dataclass(slots=True, frozen=True)
class Shape(Sequence[int]):
    dims: tuple[int, ...]

    def __getitem__(self, index: int | slice) -> Shape:
        dims = self.dims[index]
        return Shape((dims,) if isinstance(dims, int) else dims)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Shape) and self.dims == other.dims

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Shape({', '.join(map(str, self.dims))})"

    def __bool__(self) -> bool:
        return len(self.dims) > 0

    def __len__(self) -> int:
        return self.ndims

    def __iter__(self) -> Iterator[int]:
        return iter(self.dims)

    def broadcast(self, other: Shape) -> Shape:
        if self == other:
            return self
        if len(self) > len(other):
            return self.broadcast(other.lpad(len(self) - len(other)))
        if len(self) < len(other):
            return other.broadcast(self.lpad(len(other) - len(self)))
        assert all(a == b or a == 1 or b == 1 for a, b in zip(self, other)), f"Broadcast {self=} <> {other=} failed"
        return Shape(tuple(max(a, b) for a, b in zip(self, other)))

    def permute(self, axes: Sequence[int]) -> Shape:
        return Shape(tuple(self.dims[i] for i in axes))

    def swapaxes(self, ax1: int, ax2: int) -> Shape:
        axes = list(range(len(self)))
        axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
        return self.permute(axes)

    def insertaxes(self, *axes: int) -> Shape:
        new_axes = list(self)
        for i in sorted(axes, reverse=True):
            new_axes.insert(i, 1)
        return Shape(tuple(new_axes))

    def addaxes(self, idx: int, n_dims: int) -> Shape:
        return Shape(self.dims[:idx] + (1,) * n_dims + self.dims[idx:])

    def lpad(self, n_dims: int) -> Shape:
        return self.addaxes(0, n_dims)

    def rpad(self, n_dims: int) -> Shape:
        return self.addaxes(len(self), n_dims)

    def dropaxes(self, *axes: int) -> Shape:
        pos_axes = set(self.normalize_idxs(*axes))
        return Shape(tuple(d for i, d in enumerate(self) if i not in pos_axes))

    def flat(self) -> Shape:
        return Shape((self.size,))

    def normalize_idxs(self, *idxs: int) -> tuple[int, ...]:
        own_len = len(self)
        return tuple(idx % own_len if idx < 0 else idx for idx in idxs)

    @property
    def ndims(self) -> int:
        return len(self.dims)

    @property
    def size(self) -> int:
        return math.prod(self.dims)

    @classmethod
    def from_data(cls, data: llops.PyArrayRepr, /) -> Shape:
        assert isinstance(data, (int, float, bool, Sequence)), f"Unknown {data=}"
        return cls(tuple((len(data), *cls.from_data(data[0])))) if isinstance(data, Sequence) else cls(())
