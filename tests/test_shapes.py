from typing import Sequence

import pytest

from skinnygrad import shapes


def test_shape_eq():
    assert shapes.Shape((2, 3)) == shapes.Shape((2, 3))
    assert shapes.Shape((1,)) != shapes.Shape((1, 2))


def test_shape_repr():
    assert repr(shapes.Shape((2, 3))) == "Shape(2, 3)"
    assert str(shapes.Shape((4, 5, 6))) == "Shape(4, 5, 6)"


def test_shape_len():
    assert len(shapes.Shape((2, 3, 4))) == 3
    assert len(shapes.Shape(())) == 0


def test_shape_iter():
    assert list(iter(shapes.Shape((2, 3, 4)))) == [2, 3, 4]
    assert list(iter(shapes.Shape(()))) == []


def test_shape_lpad():
    assert shapes.Shape((3, 4)).lpad(2) == shapes.Shape((1, 1, 3, 4))
    assert shapes.Shape((3,)).lpad(1) == shapes.Shape((1, 3))


def test_shape_flat():
    assert shapes.Shape((2, 3)).flat() == shapes.Shape((6,))
    assert shapes.Shape(()).flat() == shapes.Shape((1,))


def test_shape_size():
    assert shapes.Shape((2, 3, 4)).size == 24
    assert shapes.Shape(()).size == 1


def test_shape_from_data():
    assert shapes.Shape.from_data([1, 2, 3]) == shapes.Shape((3,))
    assert shapes.Shape.from_data([[1, 2], [3, 4]]) == shapes.Shape((2, 2))


@pytest.mark.parametrize(
    "shape1, shape2, expected_broadcast",
    [
        ((3, 1), (1, 4), (3, 4)),
        ((2, 3, 1), (1, 1, 5), (2, 3, 5)),
        ((1,), (3,), (3,)),
    ],
)
def test_shape_broadcast(shape1, shape2, expected_broadcast):
    assert shapes.Shape(shape1).broadcast(shapes.Shape(shape2)) == shapes.Shape(expected_broadcast)


@pytest.mark.parametrize("shape1, shape2", [((2,), (3,))])
def test_shape_broadcast_failure(shape1, shape2):
    with pytest.raises(AssertionError):
        shapes.Shape(shape1).broadcast(shapes.Shape(shape2))


@pytest.mark.parametrize(
    "original_shape, slice_params, expected_shape",
    [
        ((10, 20, 30), [(2, 5), (10, 15), (5, 10)], (3, 5, 5)),
        ((10, 20, 30), [(-8, -5), (-10, -5), (-25, -20)], (3, 5, 5)),
        ((10, 20, 30), [(None, 5), (10, None), (None, None)], (5, 10, 30)),
        ((10, 20, 30), [(None, None), (None, None), (None, None)], (10, 20, 30)),
        ((10, 20, 30), [(0, 1), (0, 1), (0, 1)], (1, 1, 1)),
        ((10, 20, 30), [(0, 10), (0, 20), (0, 30)], (10, 20, 30)),
        ((10, 20, 30), [(-10, -5), (15, None), (None, 10)], (5, 5, 10)),
        ((10, 20, 30), [..., (5, 10)], (10, 20, 5)),
        ((10, 20, 30), [5, ..., 10], (20,)),
        ((10, 20, 30), [(2, 5), ...], (3, 20, 30)),
        ((10, 20, 30), [(2, 5), 1, ...], (3, 30)),
        ((10, 20, 30), [None, (0, 5), ..., None], (10, 5, 30)),
        ((10, 20, 30), [(2, 5), None, (10, 15)], (3, 20, 5)),
        ((10, 20, 30), [2, None, ..., 10], (20,)),
        ((10, 20, 30), [2, 1, (5, 10)], (5,)),
    ],
)
def test_shape_slice(
    original_shape: tuple[int, ...],
    slice_params: Sequence[shapes.Loc],
    expected_shape: tuple[int, ...],
) -> None:
    assert shapes.Shape(original_shape).slice(*slice_params) == shapes.Shape(expected_shape)


@pytest.mark.parametrize(
    "original_shape, slice_params",
    [
        ((10, 20, 30), [(5, 2), (0, 20), (0, 30)]),  # Start greater than end
        ((10, 20, 30), [(0, 11), (0, 20), (0, 30)]),  # Slice beyond bounds in one dimension
        ((10, 20, 30), [(0, 10), (0, 21), (0, 30)]),  # Slice beyond bounds in another dimension
        ((10, 20, 30), [(0, 10), (0, 20), (0, 31)]),  # Slice beyond bounds in yet another dimension
        ((10, 20, 30), [(None, 30), (0, 20), (0, 30)]),  # None for start but end beyond bounds
        ((10, 20, 30), [(0, 10), (None, 30), (0, 30)]),  # Mixing valid and invalid slices
        ((10, 20, 30), [..., ..., 0]),  # Too many ellipsis
        ((10, 20, 30), [0, 0, ..., 0, 0]),  # Too many slices with ellipsis
    ],
)
def test_shape_slice_error(original_shape, slice_params):
    with pytest.raises(AssertionError):
        shapes.Shape(original_shape).slice(*slice_params)


@pytest.mark.parametrize(
    "original_shape, pad_params, expected_shape",
    [
        ((10, 20, 30), [(1, 1), (2, 2), (3, 3)], (12, 24, 36)),
        ((10, 20, 30), [(0, 0), (0, 0), (0, 0)], (10, 20, 30)),
        ((10, 20, 30), [(5, 5), (0, 0), (2, 3)], (20, 20, 35)),
        ((0, 20, 30), [(1, 1), (1, 1), (1, 1)], (2, 22, 32)),
        ((10, 0, 30), [(3, 2), (4, 3), (5, 5)], (15, 7, 40)),
        ((10,), [(2, 3)], (15,)),
        ((0,), [(1, 1)], (2,)),
        ((5, 10), [(0, 0), (3, 4)], (5, 17)),
        ((5,), [(1, 0)], (6,)),
        ((10, 10), [(100, 100), (100, 100)], (210, 210)),
    ],
)
def test_shape_pad(
    original_shape: tuple[int, ...],
    pad_params: list[tuple[int, int]],
    expected_shape: tuple[int, ...],
) -> None:
    assert shapes.Shape(original_shape).pad(*pad_params) == shapes.Shape(expected_shape)


@pytest.mark.parametrize(
    "original_shape, pad_params",
    [
        ((10, 20), [(1, 1)]),  # Not enough padding pairs
        ((10,), [(1, 1), (2, 2)]),  # Too many padding pairs
        ((10, 20), [(1, -1), (0, 0)]),  # Negative padding is invalid
        ((10, 20), [(1, 1), (-1, 0)]),  # Negative padding is invalid
        ((10, 20), [(1.5, 1), (1, 1)]),  # Non-integer padding is invalid
    ],
)
def test_shape_pad_error(original_shape, pad_params):
    with pytest.raises((ValueError, AssertionError)):
        shapes.Shape(original_shape).pad(*pad_params)
