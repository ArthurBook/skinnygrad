"""
GPU acceleration
"""

import logging

from cupy_engine.cupy_engine import CuPyEngine


__all__ = ["CuPyEngine"]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    return logger


import skinnygrad

skinnygrad.Configuration(engine=CuPyEngine())
logger = setup_logger()
logger.info("%s set as skinnygrad runtime", CuPyEngine.__name__)
