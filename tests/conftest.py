from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from skinnygrad import runtime

if TYPE_CHECKING:
    from _pytest.python import Metafunc

ENGINES = [runtime.NumPyEngine]


def pytest_generate_tests(metafunc: Metafunc) -> None:
    if engine.__name__ in metafunc.fixturenames:
        metafunc.parametrize(engine.__name__, ENGINES, indirect=True)


@pytest.fixture
def engine(request: pytest.FixtureRequest) -> object:
    if request.param == runtime.NumPyEngine:
        return runtime.NumPyEngine()
    raise ValueError("invalid internal test config")
