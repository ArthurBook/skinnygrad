"""
CallBacks for the autodiff package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

CallbackType = TypeVar("CallbackType", bound="Callback")


if TYPE_CHECKING:
    from autodiff import llops


@runtime_checkable
class Callback(Protocol): ...


### Callbacks ###
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
