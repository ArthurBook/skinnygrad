"""
Shape tracking
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import types
from typing import TYPE_CHECKING, Iterator, Sequence

if TYPE_CHECKING:
    from skinnygrad import llops

Loc = None | int | types.EllipsisType | tuple[None | int, None | int]


@dataclasses.dataclass(slots=True, frozen=True)
class Shape:
    dims: tuple[int, ...]

    def slice(self, *locs: Loc, _skip_norm: bool = False) -> Shape:
        locs = locs if _skip_norm else self.normalize_loc(locs)
        slice_dims = (loc for loc in locs if isinstance(loc, tuple))
        return Shape(tuple(end - start for start, end in slice_dims))  # type: ignore

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

    def pad(self, *pad_per_dim: tuple[int, int]) -> Shape:
        assert len(pad_per_dim) == len(self), f"{len(pad_per_dim)=} != {len(self)=}"
        assert all(len(p) == 2 for p in pad_per_dim), f"{pad_per_dim=} is not a sequence of pairs"
        assert all(all(isinstance(i, int) for i in p) for p in pad_per_dim), f"{pad_per_dim=} contains non-ints"
        assert all(all(i >= 0 for i in p) for p in pad_per_dim), f"{pad_per_dim=} contains negative ints"
        return Shape(tuple(d + l + r for (l, r), d in zip(pad_per_dim, self)))

    def lpad(self, n_dims: int) -> Shape:
        return self.addaxes(0, n_dims)

    def rpad(self, n_dims: int) -> Shape:
        return self.addaxes(len(self), n_dims)

    def dropaxes(self, *axes: int) -> Shape:
        pos_axes = set(self.normalize_dim_ref(*axes))
        return Shape(tuple(d for i, d in enumerate(self) if i not in pos_axes))

    def flat(self) -> Shape:
        return Shape((self.size,))

    def normalize_loc(self, locs: Sequence[Loc]) -> tuple[int | tuple[int, int], ...]:
        assert (n_ellpises := (locs := list(locs)).count(Ellipsis)) <= 1, f"too many ellipses in {locs=}"
        assert (nlocs := len(locs) - n_ellpises) <= (ndims := self.ndims), f"more {nlocs=} than {ndims=}"
        assert (n_pads := ndims - nlocs) >= 0, f"{nlocs=} - {n_ellpises=} > {ndims=}"
        pad_loc = locs.index(Ellipsis) if n_ellpises else ndims
        locs = locs[:pad_loc] + [None] * n_pads + locs[pad_loc + 1 :]

        def normalize_slice(dim_i: int, slice_: Loc) -> int | tuple[int, int]:
            match slice_:
                case int(idx_slice):
                    return self.normalize_dim_slice_idx(dim_i, idx_slice)
                case None:
                    return (0, self.dims[dim_i])
                case (start, end):
                    start = self.normalize_dim_slice_idx(dim_i, start, default=0)
                    end = self.normalize_dim_slice_idx(dim_i, end, default=self.dims[dim_i])
                    assert 0 <= start < end <= self.dims[dim_i], f"{locs[dim_i]=} out of bounds for {self.dims[dim_i]=}"
                    return (start, end)
                case _:
                    raise ValueError(f"Unrecognized position type: {locs[dim_i]=}")

        assert len(locs) == self.ndims, f"{len(locs)=} != {self.ndims=}"
        return tuple(itertools.starmap(normalize_slice, enumerate(locs)))

    def normalize_dim_slice_idx(self, dim: int, slice_: int | None, /, default: int | None = None) -> int:
        (dim,) = self.normalize_dim_ref(dim)
        if slice_ is None:
            assert default is not None, f"{slice_=} is None and {default=} is None"
            return default
        slice_ = slice_ % self.dims[dim] if slice_ < 0 else slice_
        assert 0 <= slice_ <= self.dims[dim], f"{slice_=} out of bounds for {self.dims[dim]=}"
        return slice_

    def normalize_dim_ref(self, *idxs: int) -> tuple[int, ...]:
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
