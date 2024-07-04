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
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ParamSpec,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    ## optional extensions
    import visualization

    from autodiff import llops, runtime

CallbackType = TypeVar("CallbackType", bound="Callback")
CallbackSignature = ParamSpec("CallbackSignature")


### Callbacks ###
@runtime_checkable
class Callback(Protocol): ...


class CallbackStack:
    def __init__(self, *callback_types: type[Callback]) -> None:
        self._callbacks = {t: [] for t in callback_types}

    def __getitem__(self, callback_type: type[CallbackType]) -> list[CallbackType]:
        return self._callbacks[callback_type]

    def insert_callbacks(self, *callback_maybe: Callback | Any) -> None:
        for cb in callback_maybe:
            for cb_type, cb_stack in self._callbacks.items():
                if isinstance(cb, cb_type):
                    cb_stack.insert(0, cb)

    def drop_callbacks(self, *callback_maybe: Any) -> None:
        """Drop callbacks for the autodiff package."""
        for cb in callback_maybe:
            for cb_type, cb_stack in self._callbacks.items():
                if isinstance(cb, cb_type):
                    cb_stack.remove(cb)


class OnSymbolInitCallBack(Callback):
    def on_symbol_creation(self, symbol: llops.Symbol) -> None: ...
class OnCtxExitCallBack(Callback):
    def on_ctx_exit(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


class _StackedConfigMeta(type):
    __singletons__: ClassVar[dict[Type, Any]] = {}
    __regular_attrs__: set[str]  # attrs that are not accessed through context stack
    __context_stack__: collections.ChainMap[str, Any]

    def __call__(cls, **config_dict: Any) -> Configuration:
        assert not (kwargs := {k: v for k, v in config_dict.items() if v is None}), f"{kwargs=}"
        if cls in cls.__singletons__:  # update the singleton config context
            cls.__context_stack__.maps.insert(0, config_dict)
            cls.__init__(cls, **config_dict)  # __init__ is called every time the context is entered
        else:  # initialize the singleton config and ctx stack that holds all attrs
            cls.__context_stack__ = collections.ChainMap(config_dict)
            cls.__setattribute__ = cls.__context_stack__.__setitem__
            cls.__regular_attrs__ = set(dir(cls)) - set(dir(_StackedConfigMeta))
            cls.__singletons__[cls] = super().__call__(**config_dict)
        return cls.__singletons__[cls]

    def __getattr__(cls: type[_StackedConfigMeta], key: str):
        if key in cls.__regular_attrs__:
            return super().__getattribute__(key)
        return cls.__context_stack__[key]


class Configuration(contextlib.ContextDecorator, metaclass=_StackedConfigMeta):
    """Configuration for the autodiff package."""

    engine: runtime.Engine
    callback_stack = CallbackStack(OnSymbolInitCallBack, OnCtxExitCallBack)

    def __init__(
        self,
        *,
        engine: runtime.Engine | None = None,
        visualizer: visualization.GraphVisualizer | None = None,
    ) -> None:
        self.callback_stack.insert_callbacks(engine, visualizer)

    @classmethod
    def on_symbol_creation(cls, symbol: llops.Symbol) -> None:
        for callback in cls.callback_stack[OnSymbolInitCallBack]:
            callback.on_symbol_creation(symbol)

    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for exit_callback in self.callback_stack[OnCtxExitCallBack]:
            exit_callback.on_ctx_exit(exc_type, exc_value, traceback)
        dropped_context = self.__context_stack__.maps.pop(0)
        self.callback_stack.drop_callbacks(*dropped_context.values())
