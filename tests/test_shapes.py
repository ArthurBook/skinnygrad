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
