from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
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


@pytest.fixture(autouse=True)
def set_random_seeds(seed: float = 42):
    np.random.seed(seed)
    random.seed(seed)
