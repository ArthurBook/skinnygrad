import contextlib
import os
from autodiff import callbacks
from autodiff.config import Configuration
from autodiff.runtime import Engine, NumPyEngine
from autodiff.tensors import Tensor

## extension_callbacks ##
CALLBACKS: list[callbacks.Callback] = []

if bool(os.getenv(EAGER_EXECUTION_ENV_VAR := "TINYDIFF_EAGEREXECUTION", False)):
    Configuration(callbacks.EagerExecution())

with contextlib.suppress(ImportError):
    import logging_callback

    Configuration(logging_callback.TinyDiffLogger())


### Default configuration for the autodiff engine ###
Configuration(engine=NumPyEngine())


__all__ = ["Engine", "NumPyEngine", "Configuration", "Tensor"]
