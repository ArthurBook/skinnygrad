import logging
import os

from skinnygrad import callbacks, llops

LOG_LEVEL_ENV_SETTER = "SKINNYGRAD_LOGLEVEL"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging._nameToLevel[os.environ.get(LOG_LEVEL_ENV_SETTER, "INFO")])
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-10s%(funcName)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)
    return logger


default_logger = setup_logger()


class SkinnyGradLogger(callbacks.OnSymbolInitCallBack):
    def __init__(self, logger: logging.Logger = default_logger) -> None:
        super().__init__()
        self._logger = logger

    def on_symbol_creation(self, symbol: llops.Symbol) -> None:
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.log(
                logging.DEBUG,
                "%(op)-10s(%(in shapes)-20s) → %(out shape)s",
                {
                    "in shapes": ", ".join(map(str, (s.shape.dims for s in symbol.symbol_args.values()))),
                    "op": symbol.op.name,
                    "out shape": repr(symbol.shape),
                },
            )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(verbosity={self._logger.level})"
