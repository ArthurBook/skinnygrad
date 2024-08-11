"""
Logging
"""

from logging_callback.logging_callback import SkinnyGradLogger, default_logger

__all__ = ["SkinnyGradLogger"]


import skinnygrad

skinnygrad.Configuration(logger := SkinnyGradLogger())
default_logger.info("%s set as logger for skinnygrad", str(logger))
