"""
GPU acceleration
"""

import logging

from cupy_engine.cupy_engine import CuPyEngine


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)  # Use __name__ to get the module name
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    return logger


__all__ = ["CuPyEngine"]

import skinnygrad

skinnygrad.Configuration(engine=CuPyEngine())

## inform user that the CuPyEngine is set as the engine for skinnygrad.
logger = setup_logger()
logger.info("%s set as skinnygrad runtime", CuPyEngine.__name__)
