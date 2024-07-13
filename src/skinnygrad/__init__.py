import contextlib
import os

from skinnygrad import callbacks
from skinnygrad.config import Configuration
from skinnygrad.runtime import Engine, NumPyEngine
from skinnygrad.tensors import Tensor

## extension_callbacks ##
CALLBACKS: list[callbacks.Callback] = []

if bool(os.getenv(EAGER_EXECUTION_ENV_VAR := "SKINNYGRAD_EAGEREXECUTION", False)):
    Configuration(callbacks.EagerExecution())

with contextlib.suppress(ImportError):
    import logging_callback

    Configuration(logging_callback.SkinnyGradLogger())


### Default configuration ###
Configuration(engine=NumPyEngine())


__all__ = ["Engine", "NumPyEngine", "Configuration", "Tensor"]
