"""
Callbacks for the skinnygrad package.
"""

from __future__ import annotations

import abc
import collections
import types
from typing import TYPE_CHECKING, Any, Iterable, TypeVar

CallbackType = TypeVar("CallbackType", bound="Callback")


if TYPE_CHECKING:
    from skinnygrad import llops


class Callback(abc.ABC): ...


### Callbacks ###
class CallbackStack:
    def __init__(self, callbacks_maybe: Iterable[Callback | Any]) -> None:
        self._callbacks = {cb: collections.deque[Callback]() for cb in Callback.__subclasses__()}
        self.insert_callbacks(*callbacks_maybe)

    def __getitem__(self, callback_type: type[CallbackType]) -> list[CallbackType]:
        return self._callbacks[callback_type]  # type: ignore

    def insert_callbacks(self, *callback_maybe: Callback | Any) -> None:
        for cb in callback_maybe:
            for cb_type, cb_stack in self._callbacks.items():
                if isinstance(cb, cb_type):
                    cb_stack.insert(0, cb)

    def drop_callbacks(self, *callback_maybe: Any) -> None:
        for cb in callback_maybe:
            for cb_type, cb_stack in self._callbacks.items():
                if isinstance(cb, cb_type):
                    cb_stack.remove(cb)


class OnSymbolInitCallBack(Callback, abc.ABC):
    def on_symbol_creation(self, symbol: llops.Symbol) -> None: ...
class OnCtxEnterCallBack(Callback, abc.ABC):
    def on_ctx_enter(self) -> None: ...
class OnCtxExitCallBack(Callback, abc.ABC):
    def on_ctx_exit(self, exc_type: type[Exception], exc_value: Exception, traceback: types.TracebackType) -> None: ...


### Builtin callbacks ###
class EagerExecution(OnSymbolInitCallBack):
    def on_symbol_creation(self, symbol: llops.Symbol) -> None:
        symbol.realize()
