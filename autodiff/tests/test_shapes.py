import pytest

from autodiff.llops import Shape


def test_shape_eq():
    assert Shape((2, 3)) == Shape((2, 3))
    assert Shape((1,)) != Shape((1, 2))


def test_shape_repr():
    assert repr(Shape((2, 3))) == "Shape(2, 3)"
    assert str(Shape((4, 5, 6))) == "Shape(4, 5, 6)"


def test_shape_len():
    assert len(Shape((2, 3, 4))) == 3
    assert len(Shape(())) == 0


def test_shape_iter():
    assert list(iter(Shape((2, 3, 4)))) == [2, 3, 4]
    assert list(iter(Shape(()))) == []


def test_shape_lpad():
    assert Shape((3, 4)).lpad(2) == Shape((1, 1, 3, 4))
    assert Shape((3,)).lpad(1) == Shape((1, 3))


def test_shape_flat():
    assert Shape((2, 3)).flat() == Shape((6,))
    assert Shape(()).flat() == Shape((1,))


def test_shape_size():
    assert Shape((2, 3, 4)).size == 24
    assert Shape(()).size == 1


def test_shape_from_data():
    assert Shape.from_data([1, 2, 3]) == Shape((3,))
    assert Shape.from_data([[1, 2], [3, 4]]) == Shape((2, 2))


@pytest.mark.parametrize(
    "shape1, shape2, expected_broadcast",
    [
        ((3, 1), (1, 4), (3, 4)),
        ((2, 3, 1), (1, 1, 5), (2, 3, 5)),
        ((1,), (3,), (3,)),
    ],
)
def test_shape_broadcast(shape1, shape2, expected_broadcast):
    assert Shape(shape1).broadcast(Shape(shape2)) == Shape(expected_broadcast)


@pytest.mark.parametrize("shape1, shape2", [((2,), (3,))])
def test_shape_broadcast_failure(shape1, shape2):
    with pytest.raises(AssertionError):
        Shape(shape1).broadcast(Shape(shape2))
