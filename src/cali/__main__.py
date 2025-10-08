"""Module to run the cali application."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from cali import CellposeBatchSegmentation, PlateViewer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

WALLE_ICON = Path(__file__).parent / "icons" / "wall_e_icon.png"
CELLPOSE_ICON = Path(__file__).parent / "icons" / "cellpose_icon.png"


def main(args: Sequence[str] | None = None) -> None:
    """Run the cali application."""
    from fonticon_mdi6 import MDI6
    from qtpy.QtGui import QIcon
    from superqt.fonticon import icon

    app = QApplication([])
    app.setWindowIcon(QIcon(icon(MDI6.view_comfy, color="#00FF00")))
    pl = PlateViewer()
    pl.show()
    sys.excepthook = _our_excepthook
    app.exec()


def batch_cellpose() -> None:
    """Open the Batch Cellpose Segmentation."""
    app = QApplication([])
    app.setWindowIcon(QIcon(str(CELLPOSE_ICON)))
    cp = CellposeBatchSegmentation()
    cp.show()
    sys.excepthook = _our_excepthook
    app.exec()


def _our_excepthook(
    type: type[BaseException], value: BaseException, tb: TracebackType | None
) -> None:
    """Excepthook that prints the traceback to the console.

    By default, Qt's excepthook raises sys.exit(), which is not what we want.
    """
    # this could be elaborated to do all kinds of things...
    traceback.print_exception(type, value, tb)


if __name__ == "__main__":
    main()
