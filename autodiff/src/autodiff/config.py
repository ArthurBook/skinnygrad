import contextlib
import threading
from typing import ClassVar, TypeVar

from autodiff import runtime
from autodiff.engines.packaged_engines import numpy_engine

GLOBAL_CTX_LOCK = threading.Lock()


T = TypeVar("T")


class UNSET: ...


class Configuration(contextlib.ContextDecorator):
    backend: ClassVar[runtime.Engine]

    def __init__(self, backend: runtime.Engine | UNSET = UNSET()) -> None:
        Configuration.backend = backend  # type: ignore

    def __enter__(self):
        return  ## TODO lock thread here in a way that GIL steps through this all the way first

    def __exit__(self, a, b, c):
        return


###
# Default config
###
Configuration(
    backend=numpy_engine.NUMPY_ENGINE,
)
