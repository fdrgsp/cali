from importlib.metadata import PackageNotFoundError, version
from ._plate_viewer import PlateViewer

try:
    __version__ = version("micromanager-gui")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Federico Gasparoli"
__email__ = "federico.gasparoli@gmail.com"

__all__ = ["PlateViewer"]
