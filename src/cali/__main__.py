"""Module to run the cali application."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from superqt import QIconifyIcon

# CRITICAL: Import torch before Qt on Windows to avoid DLL conflicts
# When PyQt6 initializes before PyTorch on Windows, it can cause c10.dll failures
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        # Cellpose is optional, so torch might not be installed
        pass

from qtpy.QtWidgets import QApplication
from useq import register_well_plates

from cali.gui import CaliGui

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType


CELLPOSE_ICON = Path(__file__).parent / "icons" / "cellpose_icon.png"


# Register more useq-plates
register_well_plates(
    {
        "1536-well": {
            "rows": 32,
            "columns": 48,
            "well_spacing": 2.25,
            "well_size": 1.55,
        },
        "dish-35mm-round": {
            "rows": 1,
            "columns": 1,
            "well_spacing": 0.0,
            "well_size": 35.0,
            "circular_wells": True,
            "name": "dish-35mm-round",
        },
        "coverslip-18mm-round": {
            "rows": 1,
            "columns": 1,
            "well_spacing": 0.0,
            "well_size": 18.0,
            "circular_wells": True,
            "name": "coverslip-18mm-round",
        },
        "coverslip-22mm-round": {
            "rows": 1,
            "columns": 1,
            "well_spacing": 0.0,
            "well_size": 22.0,
            "circular_wells": True,
            "name": "coverslip-22mm-round",
        },
        # overvriting because in useq name and key are different and we need them same
        "coverslip-18mm-square": {
            "rows": 1,
            "columns": 1,
            "well_spacing": 0.0,
            "well_size": 18.0,
            "circular_wells": False,
            "name": "coverslip-18mm-square",
        },
        # overvriting because in useq name and key are different and we need them same
        "coverslip-22mm-square": {
            "rows": 1,
            "columns": 1,
            "well_spacing": 0.0,
            "well_size": 22.0,
            "circular_wells": False,
            "name": "coverslip-22mm-square",
        },
    }
)


def main(args: Sequence[str] | None = None) -> None:
    """Run the cali application."""
    from cali.logger import cali_logger, set_console_level

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cali - Calcium Imaging Analysis")
    parser.add_argument(
        "--logger",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parsed_args = parser.parse_args(args)

    # Set logger level
    log_level = getattr(logging, parsed_args.logger.upper())
    cali_logger.setLevel(log_level)
    set_console_level(log_level)  # Also set console handler level
    cali_logger.info(f"Logger level set to {parsed_args.logger.upper()}")

    app = QApplication([])
    app.setWindowIcon(QIconifyIcon("mdi:view-comfy", color="#00FF00"))
    pl = CaliGui()
    pl.show()
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
