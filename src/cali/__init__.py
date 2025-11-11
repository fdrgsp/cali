"""Cali package."""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version

# need to suppress the cvxpy warning from oasis before importing cali
warnings.filterwarnings("ignore", "Could not find cvxpy.*", UserWarning)

try:
    __version__ = version("micromanager-gui")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Federico Gasparoli"
__email__ = "federico.gasparoli@gmail.com"
