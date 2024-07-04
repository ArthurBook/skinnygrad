import logging
import os

from autodiff import callbacks, llops

LOG_LEVEL_ENV_SETTER = "TINYDIFF_LOGLEVEL"

logging.basicConfig(
    level=logging._nameToLevel[os.environ.get(LOG_LEVEL_ENV_SETTER, "INFO")],
    format="%(asctime)s %(levelname)-9s %(funcName)s: - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

default_logger = logging.getLogger(__name__)


class TinyDiffLogger(callbacks.OnSymbolInitCallBack):
    def __init__(self, logger: logging.Logger = default_logger) -> None:
        super().__init__()
        self._logger = logger

    def on_symbol_creation(self, symbol: llops.Symbol) -> None:
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.log(
                logging.DEBUG,
                "%(op)-10s(%(in shapes)-20s) → %(out shape)s",
                {
                    "in shapes": ", ".join(str(s.shape.dims) for s in symbol.src),
                    "op": symbol.op.name,
                    "out shape": repr(symbol.shape),
                },
            )
