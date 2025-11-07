"""Cali package."""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

# need to suppress the cvxpy warning from oasis before importing cali
warnings.filterwarnings("ignore", "Could not find cvxpy.*", UserWarning)

from ._plate_viewer import PlateViewer  # noqa: E402

if TYPE_CHECKING:
    from ._batch_cellpose import CellposeBatchSegmentation

try:
    __version__ = version("micromanager-gui")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Federico Gasparoli"
__email__ = "federico.gasparoli@gmail.com"

__all__ = ["CellposeBatchSegmentation", "PlateViewer"]


def __getattr__(name: str) -> object:
    """Lazy import for heavy dependencies."""
    if name == "CellposeBatchSegmentation":
        from ._batch_cellpose import CellposeBatchSegmentation

        return CellposeBatchSegmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
