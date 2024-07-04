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

from autodiff import callbacks

if TYPE_CHECKING:
    from autodiff import llops, runtime


class _StackedConfigMeta(type):
    __singletons__: ClassVar[dict[Type, Any]] = {}
    __regular_attrs__: set[str]  # attrs that are not accessed through context stack
    __context_stack__: collections.ChainMap[str, Any]

    def __call__(cls, *callback: callbacks.Callback, **config_dict: Any) -> Configuration:
        assert not (kwargs := {k: v for k, v in config_dict.items() if v is None}), f"{kwargs=}"
        if cls in cls.__singletons__:  # update the singleton config context
            cls.__context_stack__.maps.insert(0, config_dict)
            cls.__init__(cls, *callback, **config_dict)  # __init__ handles callbacks
        else:  # initialize the singleton config and ctx stack that holds all attrs
            cls.__context_stack__ = collections.ChainMap(config_dict)
            cls.__setattribute__ = cls.__context_stack__.__setitem__
            cls.__regular_attrs__ = set(dir(cls)) - set(dir(_StackedConfigMeta))
            cls.__singletons__[cls] = super().__call__(*callback, **config_dict)
        return cls.__singletons__[cls]

    def __getattr__(cls: type[_StackedConfigMeta], key: str):
        if key in cls.__regular_attrs__:
            return super().__getattribute__(key)
        return cls.__context_stack__[key]


class Configuration(contextlib.ContextDecorator, metaclass=_StackedConfigMeta):
    """Configuration for the autodiff package."""

    engine: runtime.Engine
    callback_stack = callbacks.CallbackStack(
        callbacks.OnSymbolInitCallBack,
        callbacks.OnCtxExitCallBack,
    )

    def __init__(
        self,
        *callback: callbacks.Callback,
        ## runtime ctx
        engine: runtime.Engine | None = None,
        **context: Any,
    ) -> None:
        del engine, context  # kwargs were taken care of by to metaclass __call__
        self.callback_stack.insert_callbacks(*callback)

    @classmethod
    def on_symbol_creation(cls, symbol: llops.Symbol) -> None:
        for callback in cls.callback_stack[callbacks.OnSymbolInitCallBack]:
            callback.on_symbol_creation(symbol)

    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for exit_callback in self.callback_stack[callbacks.OnCtxExitCallBack]:
            exit_callback.on_ctx_exit(exc_type, exc_value, traceback)
        dropped_context = self.__context_stack__.maps.pop(0)
        self.callback_stack.drop_callbacks(*dropped_context.values())
