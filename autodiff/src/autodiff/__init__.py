from autodiff.config import Configuration
from autodiff.runtime import Engine, NumPyEngine
from autodiff.tensors import Tensor

### Default configuration for the autodiff engine ###
Configuration(
    engine=NumPyEngine(),
)


__all__ = ["Engine", "NumPyEngine", "Configuration", "Tensor"]
