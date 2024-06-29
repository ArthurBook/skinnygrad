"""
# Global configuration for the autodiff package.

User can modify this configuration in 2 ways:

## Permanent change
```python
from autodiff import config
config.Configuration(engine=engine)
```

## Temporary change
```python
from autodiff import config
with config.Configuration(engine=engine):
    ...
"""

from __future__ import annotations

import collections
import contextlib
from typing import TYPE_CHECKING, Any, ClassVar, Type

if TYPE_CHECKING:
    from autodiff import runtime


class _NOT_GIVEN: ...


class _StackedConfigMeta(type):
    __singletons__: ClassVar[dict[Type, Any]] = {}
    __context_stack__: collections.ChainMap[str, Any]

    def __call__(cls, **config_dict: Any) -> Configuration:
        assert _NOT_GIVEN not in config_dict.values()
        if cls in cls.__singletons__:  # update the singleton config context
            cls.__context_stack__.maps.insert(0, config_dict)
        else:  # initialize the singleton config and ctx stack that holds all attrs
            cls.__context_stack__ = collections.ChainMap(config_dict)
            cls.__setattribute__ = cls.__context_stack__.__setitem__
            cls.__singletons__[cls] = super().__call__()
        return cls.__singletons__[cls]

    def __getattr__(cls: type[_StackedConfigMeta], key: str):
        return cls.__context_stack__[key]


class Configuration(contextlib.ContextDecorator, metaclass=_StackedConfigMeta):
    """Configuration for the autodiff package."""

    engine: runtime.Engine

    def __init__(self, *, engine: runtime.Engine | _NOT_GIVEN = _NOT_GIVEN()) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.__context_stack__.maps.pop(0)
