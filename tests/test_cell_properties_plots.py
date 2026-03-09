"""Tests for cell properties bar plot functions.

Covers:
- compute_cell_size_data
- compute_percentage_active_data
- compute_percentage_active_stim_split_data
- plot_percentage_active_stim_split_bar_plot (empty stim data)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget
from sqlmodel import create_engine

from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
from cali.sqlmodel._util import create_database_and_tables

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Headless compute
# ---------------------------------------------------------------------------


def test_compute_cell_size_data(full_db: tuple[Engine, int]) -> None:
    from cali.plot._multi_wells_plots._cell_properties import compute_cell_size_data

    engine, run_id = full_db
    result = compute_cell_size_data(engine, run_id)
    assert result is not None
    _bar_data, name, units = result
    assert name == "Cell Size"
    assert units == "μm²"


def test_compute_cell_size_data_empty_db() -> None:
    from cali.plot._multi_wells_plots._cell_properties import compute_cell_size_data

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert compute_cell_size_data(engine, None) is None
    engine.dispose(close=True)


def test_compute_percentage_active_data(full_db: tuple[Engine, int]) -> None:
    from cali.plot._multi_wells_plots._cell_properties import (
        compute_percentage_active_data,
    )

    engine, run_id = full_db
    result = compute_percentage_active_data(engine, run_id)
    assert result is not None
    _bar_data, name, units = result
    assert name == "Percentage Active ROIs"
    assert units == "%"


def test_compute_percentage_active_data_empty_db() -> None:
    from cali.plot._multi_wells_plots._cell_properties import (
        compute_percentage_active_data,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert compute_percentage_active_data(engine, None) is None
    engine.dispose(close=True)


def test_compute_percentage_active_stim_split_data_empty_db() -> None:
    from cali.plot._multi_wells_plots._cell_properties import (
        compute_percentage_active_stim_split_data,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert compute_percentage_active_stim_split_data(engine, None) is None
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Stim-split empty data path
# ---------------------------------------------------------------------------


def test_percentage_active_stim_split_empty(qtbot: QtBot) -> None:
    """plot_percentage_active_stim_split_bar_plot handles empty stim data."""
    from cali.plot._multi_wells_plots._cell_properties import (
        plot_percentage_active_stim_split_bar_plot,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_percentage_active_stim_split_bar_plot(widget, "% Active Stim", engine, None)
    engine.dispose(close=True)
