"""Module to run the cali application."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

# CRITICAL: Import torch before Qt on Windows to avoid DLL conflicts
# When PyQt6 initializes before PyTorch on Windows, it can cause c10.dll failures
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        # Cellpose is optional, so torch might not be installed
        pass

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType


CELLPOSE_ICON = Path(__file__).parent / "icons" / "cellpose_icon.png"


def _str_to_bool(value: str) -> bool:
    """Convert a string value to bool for argparse."""
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _tree_command(parsed_args: argparse.Namespace) -> None:
    """Execute the ``tree`` sub-command."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    from cali.sqlmodel import print_cali_results

    db_path = Path(parsed_args.db).resolve()
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    # NullPool ensures connections are closed immediately (not kept in a pool).
    engine = create_engine(f"sqlite:///{db_path}", poolclass=NullPool)
    try:
        print_cali_results(
            engine,
            max_experiment_level=parsed_args.level,
            show_settings=parsed_args.show_settings,
        )
    finally:
        engine.dispose()


def main(args: Sequence[str] | None = None) -> None:
    """Run the cali application."""
    from cali.logger import cali_logger, set_console_level

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cali - Calcium Imaging Analysis")
    parser.add_argument(
        "--logger",
        "-lg",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- tree sub-command ---
    tree_parser = subparsers.add_parser(
        "tree",
        help="Print a tree view of the contents of a .cali database",
    )
    tree_parser.add_argument(
        "db",
        type=str,
        help="Path to the .cali database file",
    )
    tree_parser.add_argument(
        "--level",
        "-l",
        dest="level",
        type=str,
        default="roi",
        choices=["experiment", "plate", "well", "fov", "roi"],
        help="Maximum depth to show in the tree (default: roi)",
    )
    tree_parser.add_argument(
        "--show-settings",
        "-s",
        dest="show_settings",
        type=_str_to_bool,
        default=True,
        metavar="{True,False}",
        help="Show analysis settings in the tree (default: True)",
    )

    parsed_args = parser.parse_args(args)

    # Set logger level
    log_level = getattr(logging, parsed_args.logger.upper())
    cali_logger.setLevel(log_level)
    set_console_level(log_level)  # Also set console handler level

    if parsed_args.command == "tree":
        _tree_command(parsed_args)
        return

    cali_logger.info(f"Logger level set to {parsed_args.logger.upper()}")

    # Default: launch the GUI
    from qtpy.QtWidgets import QApplication
    from superqt import QIconifyIcon
    from useq import register_well_plates

    from cali.gui import CaliGui

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
            # overvriting because in useq name and key are different and we need same
            "coverslip-18mm-square": {
                "rows": 1,
                "columns": 1,
                "well_spacing": 0.0,
                "well_size": 18.0,
                "circular_wells": False,
                "name": "coverslip-18mm-square",
            },
            # overvriting because in useq name and key are different and we need same
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
