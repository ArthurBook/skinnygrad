import abc
import dataclasses
import enum
from typing import ClassVar, Final, Generic, Literal, Protocol, TypeVar

import helpers
import torch

###
# Types
###
ConfigType = TypeVar("ConfigType", bound="ActivationConfig", covariant=True)


###
# Activation names
###
class Activations(str, enum.Enum):
    RELU = "relu"
    RELU_AGAIN = "relu-again"


###
# Activation functions
###
@helpers.abstract_configuration
@dataclasses.dataclass
class ActivationConfig(abc.ABC, helpers.Config):
    __registry_name__: ClassVar[Activations]


@helpers.abstract_implementation
class Activation(abc.ABC, helpers.Model[ConfigType]):
    __registry_name__: ClassVar[Activations]

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize the activation function with the given configuration.
        If no configuration is provided, default values are used.
        If no sensible default values are available, raise ~exceptions.NeedsConfigurationError.
        """

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


###
# ReLU
###
class ReLUConfig(ActivationConfig):
    __registry_name__: Final[Literal[Activations.RELU]] = Activations.RELU


class ReLU(Activation[ReLUConfig]):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0)


class AnotherReluConfig(ActivationConfig):
    __registry_name__: Final[Literal[Activations.RELU_AGAIN]] = Activations.RELU_AGAIN


class AnotherReLU(Activation[AnotherReluConfig]):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0)


if __name__ == "__main__":
    relu = ReLU()
    x = torch.randn(10)
    print(relu(x))
    # tensor([
