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
import types
from typing import TYPE_CHECKING, Any, ClassVar, Type

from autodiff import callbacks

if TYPE_CHECKING:
    from autodiff import llops, runtime


class _StackedConfigMeta(type):
    __singletons__: ClassVar[dict[Type, Any]] = {}
    __regular_attrs__: set[str]  # attrs that are not accessed through context stack
    __context_stack__: collections.ChainMap[str, Any]  # stack of contexts that hold all attrs
    __callback_stack__: callbacks.CallbackStack  # stack of callbacks

    def __call__(cls, *callback: callbacks.Callback, **config_dict: Any) -> type[Configuration]:
        assert not (kwargs := {k: v for k, v in config_dict.items() if v is None}), f"{kwargs=}"
        if cls not in cls.__singletons__:  # initialize:=set class vars for the first time
            cls.__context_stack__ = collections.ChainMap(config_dict)
            cls.__callback_stack__ = callbacks.CallbackStack(callback)
            cls.__setattribute__ = cls.__context_stack__.__setitem__
            cls.__regular_attrs__ = set(dir(cls)) - set(dir(_StackedConfigMeta))
            cls.__singletons__[cls] = cls  # super().__call__(*callback, **config_dict)
        else:  # update the stacks wiht new contexts
            cls.__callback_stack__.insert_callbacks(*callback)
            cls.__context_stack__.maps.insert(0, config_dict)
        return cls.__singletons__[cls]

    def __getattr__(cls: type[_StackedConfigMeta], key: str):
        return super().__getattribute__(key) if key in cls.__regular_attrs__ else cls.__context_stack__[key]

    def __enter__(cls) -> None:
        for enter_callback in cls.__callback_stack__[callbacks.OnCtxEnterCallBack]:
            enter_callback.on_ctx_enter()

    def __exit__(cls, exc_type: type[Exception], exc_value: Exception, traceback: types.TracebackType) -> None:
        for exit_callback in cls.__callback_stack__[callbacks.OnCtxExitCallBack]:
            exit_callback.on_ctx_exit(exc_type, exc_value, traceback)
        dropped_context = cls.__context_stack__.maps.pop(0)
        cls.__callback_stack__.drop_callbacks(*dropped_context.values())


class Configuration(contextlib.ContextDecorator, metaclass=_StackedConfigMeta):
    """Configuration for the autodiff package."""

    engine: runtime.Engine

    def __init__(
        self,
        *callback: callbacks.Callback,
        engine: runtime.Engine | None = None,
        **context: Any,
    ) -> None: ...

    @classmethod
    def on_symbol_creation(cls, symbol: llops.Symbol) -> None:
        for callback in cls.__callback_stack__[callbacks.OnSymbolInitCallBack]:
            callback.on_symbol_creation(symbol)
