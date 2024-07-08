from autodiff import callbacks
from autodiff.config import Configuration
from autodiff.runtime import Engine, NumPyEngine
from autodiff.tensors import Tensor

extension_callbacks: list[callbacks.Callback] = []
try:
    import logging_callback

    extension_callbacks.append(logging_callback.TinyDiffLogger())
except ImportError:
    pass

### Default configuration for the autodiff engine ###
Configuration(
    *extension_callbacks,
    engine=NumPyEngine(),
)


__all__ = ["Engine", "NumPyEngine", "Configuration", "Tensor"]
