"""Integration tests for multi-well bar plots - verify actual plot data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyqtgraph import BarGraphItem
from qtpy.QtWidgets import QWidget
from sqlmodel import Session, create_engine, select

from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
from cali.plot._multi_wells_plots._calcium_peaks import (
    plot_calcium_peaks_amplitude_bar_plot,
    plot_calcium_peaks_frequency_bar_plot,
    plot_calcium_peaks_iei_bar_plot,
    plot_calcium_peaks_synchrony_bar_plot,
)
from cali.plot._multi_wells_plots._cell_properties import (
    plot_cell_size_bar_plot,
    plot_percentage_active_bar_plot,
)
from cali.plot._multi_wells_plots._spike_analysis import (
    plot_burst_avg_duration_bar_plot,
    plot_burst_avg_interval_bar_plot,
    plot_burst_count_bar_plot,
    plot_spike_synchrony_bar_plot,
)
from cali.sqlmodel import CaliResult

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def multi_well_widget_with_data(
    qtbot: QtBot,
) -> tuple[_MultilWellGraphWidget, int]:
    """Create multi-well widget with real test database."""
    db_path = "tests/test_data/multi_pos/result_2pos.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Get a valid run_id from the database
    with Session(engine) as session:
        run_id = session.exec(select(CaliResult.id).limit(1)).first()

    assert run_id is not None, "Test database should have at least one CaliResult"

    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(widget)

    widget.database_path = db_path
    widget.engine = engine
    widget.run_id = run_id

    yield widget, run_id

    engine.dispose(close=True)


def _get_bar_items(widget: _MultilWellGraphWidget) -> list[BarGraphItem]:
    """Extract BarGraphItem objects from plot widget."""
    if widget.plot_item is None:
        return []

    bar_items = []
    for item in widget.plot_item.items:
        if isinstance(item, BarGraphItem):
            bar_items.append(item)
    return bar_items


def _verify_plot_has_data(
    widget: _MultilWellGraphWidget,
    plot_name: str,
    min_bars: int = 1,
) -> None:
    """Verify that a plot has actual bar data displayed.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        The widget to check
    plot_name : str
        Name of the plot for error messages
    min_bars : int
        Minimum number of bars expected (default 1)
    """
    assert widget.plot_item is not None, f"{plot_name}: plot_item should exist"

    # Check that items were added to the plot
    assert len(widget.plot_item.items) > 0, (
        f"{plot_name}: No items in plot. The plot function may not be "
        "querying data correctly or data may be missing from test database."
    )

    # Check for BarGraphItem specifically
    bar_items = _get_bar_items(widget)
    assert len(bar_items) > 0, (
        f"{plot_name}: No BarGraphItem found in plot. "
        f"Found {len(widget.plot_item.items)} items but none are BarGraphItem."
    )

    # Check that bars have actual data
    for bar_item in bar_items:
        opts = bar_item.opts
        if "height" in opts:
            heights = opts["height"]
            if hasattr(heights, "__len__"):
                assert len(heights) >= min_bars, (
                    f"{plot_name}: BarGraphItem has {len(heights)} bars, "
                    f"expected at least {min_bars}"
                )
                # Verify at least one bar has non-zero height
                assert any(h != 0 for h in heights), (
                    f"{plot_name}: All bar heights are zero"
                )


def test_plot_calcium_peaks_amplitude_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that calcium peaks amplitude plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_calcium_peaks_amplitude_bar_plot(
        widget, "Calcium Peaks Amplitude", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Calcium Peaks Amplitude")


def test_plot_calcium_peaks_frequency_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that calcium peaks frequency plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_calcium_peaks_frequency_bar_plot(
        widget, "Calcium Peaks Frequency", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Calcium Peaks Frequency")


def test_plot_calcium_peaks_iei_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that calcium peaks IEI plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_calcium_peaks_iei_bar_plot(widget, "Calcium Peaks IEI", widget.engine, run_id)

    _verify_plot_has_data(widget, "Calcium Peaks IEI")


def test_plot_calcium_peaks_synchrony_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that calcium peaks synchrony plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_calcium_peaks_synchrony_bar_plot(
        widget, "Calcium Peaks Synchrony", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Calcium Peaks Synchrony")


def test_plot_spike_synchrony_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that spike synchrony plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_spike_synchrony_bar_plot(widget, "Spike Synchrony", widget.engine, run_id)

    _verify_plot_has_data(widget, "Spike Synchrony")


# NOTE: plot_calcium_network_density_bar_plot function doesn't exist
# def test_plot_calcium_network_density_has_data(
#     multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
# ) -> None:
#     """Test that calcium network density plot displays actual data."""
#     widget, run_id = multi_well_widget_with_data
#
#     plot_calcium_network_density_bar_plot(
#         widget, "Calcium Network Density", widget.engine, run_id
#     )
#
#     _verify_plot_has_data(widget, "Calcium Network Density")


def test_plot_burst_count_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that burst count plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_burst_count_bar_plot(widget, "Burst Count", widget.engine, run_id)

    _verify_plot_has_data(widget, "Burst Count")


def test_plot_burst_avg_duration_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that burst average duration plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_burst_avg_duration_bar_plot(
        widget, "Burst Avg Duration", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Burst Avg Duration")


def test_plot_burst_avg_interval_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that burst average interval plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_burst_avg_interval_bar_plot(
        widget, "Burst Avg Interval", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Burst Avg Interval")


def test_plot_percentage_active_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that percentage active ROIs plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_percentage_active_bar_plot(
        widget, "Percentage Active ROIs", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Percentage Active ROIs")


def test_plot_cell_size_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that cell size plot displays actual data."""
    widget, run_id = multi_well_widget_with_data

    plot_cell_size_bar_plot(widget, "Cell Size", widget.engine, run_id)

    _verify_plot_has_data(widget, "Cell Size")
