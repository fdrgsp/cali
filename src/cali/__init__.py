"""Cali package."""

import warnings
from importlib.metadata import PackageNotFoundError, version

# need to suppress the cvxpy warning from oasis before importing cali
warnings.filterwarnings("ignore", "Could not find cvxpy.*", UserWarning)

from ._batch_cellpose import CellposeBatchSegmentation  # noqa: E402
from ._plate_viewer import PlateViewer  # noqa: E402

try:
    __version__ = version("micromanager-gui")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Federico Gasparoli"
__email__ = "federico.gasparoli@gmail.com"

__all__ = ["CellposeBatchSegmentation", "PlateViewer"]
