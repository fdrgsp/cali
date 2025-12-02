"""Initialization for cali_logger."""

from ._logger import LOGGER as cali_logger
from ._logger import set_console_level

__all__ = ["cali_logger", "set_console_level"]
