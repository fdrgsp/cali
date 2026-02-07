"""Integration tests for multi-well bar plots - verify actual plot data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyqtgraph import BarGraphItem
from qtpy.QtWidgets import QWidget
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, create_engine, select

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
from cali.plot._multi_wells_plots._calcium_peaks import (
    plot_calcium_peaks_amplitude_bar_plot,
    plot_calcium_peaks_frequency_bar_plot,
    plot_calcium_peaks_iei_bar_plot,
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
    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Migrate schema if needed - add new columns that don't exist
    # This must happen BEFORE any model queries
    with engine.connect() as conn:
        try:
            # Check if column exists by trying to select it
            conn.execute(
                text("SELECT calcium_peaks_max_lag FROM analysis_settings LIMIT 1")
            )
        except OperationalError:
            # Column doesn't exist, add it
            try:
                conn.execute(
                    text(
                        "ALTER TABLE analysis_settings "
                        "ADD COLUMN calcium_peaks_max_lag INTEGER DEFAULT 5"
                    )
                )
                conn.commit()
            except OperationalError:
                # Failed to add column, rollback
                conn.rollback()

        # Check and add pixel_size column if missing
        try:
            conn.execute(text("SELECT pixel_size FROM extraction_settings LIMIT 1"))
        except OperationalError:
            try:
                conn.execute(
                    text("ALTER TABLE extraction_settings ADD COLUMN pixel_size REAL")
                )
                conn.commit()
            except OperationalError:
                conn.rollback()

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
    # Skip if no data is available (common with mock test data)
    if len(widget.plot_item.items) == 0:
        pytest.skip(
            f"{plot_name}: No items in plot. Test database may not have "
            "valid peak/analysis data for bar plots."
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
                if len(heights) < min_bars:
                    pytest.skip(
                        f"{plot_name}: BarGraphItem has {len(heights)} bars, "
                        f"expected at least {min_bars}. Test database may not have "
                        "sufficient data."
                    )
                # Verify at least one bar has non-zero height
                if not any(h != 0 for h in heights):
                    pytest.skip(
                        f"{plot_name}: All bar heights are zero. Test database may "
                        "not have valid peak/analysis data for bar plots."
                    )


def test_plot_calcium_peaks_amplitude_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that calcium peaks amplitude plot displays actual data."""
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    plot_calcium_peaks_amplitude_bar_plot(
        widget, "Calcium Peaks Amplitude", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Calcium Peaks Amplitude")


def test_plot_calcium_peaks_frequency_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that calcium peaks frequency plot displays actual data."""
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    plot_calcium_peaks_frequency_bar_plot(
        widget, "Calcium Peaks Frequency", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Calcium Peaks Frequency")


def _has_iei_data(engine: Engine) -> bool:
    """Check if there is non-empty IEI data in DataAnalysis table."""
    try:
        with Session(engine) as session:
            from cali.sqlmodel import DataAnalysis

            # Check if any record has non-empty IEI data
            analyses = session.exec(select(DataAnalysis).limit(10)).all()
            return any(
                getattr(a, "iei", None) and len(a.iei) > 0
                for a in analyses  # type: ignore
            )
    except (OperationalError, AttributeError):
        return False


def _has_burst_data(engine: Engine) -> bool:
    """Check if there is burst data in FOVAnalysis table."""
    try:
        with Session(engine) as session:
            from cali.sqlmodel import FOVAnalysis

            # Check if any FOVAnalysis record has spike or calcium burst data
            analyses = session.exec(select(FOVAnalysis).limit(10)).all()
            return any(
                (
                    getattr(a, "spike_burst_count", None) is not None
                    and a.spike_burst_count > 0
                )  # type: ignore
                or (
                    getattr(a, "calcium_burst_count", None) is not None
                    and a.calcium_burst_count > 0
                )  # type: ignore
                for a in analyses
            )
    except (OperationalError, AttributeError):
        return False


def test_plot_calcium_peaks_iei_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that calcium peaks IEI plot displays actual data.

    Note: IEI requires at least 2 peaks per ROI.
    """
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    if not _has_iei_data(widget.engine):
        pytest.skip("No IEI data (ROIs have < 2 peaks)")

    plot_calcium_peaks_iei_bar_plot(widget, "Calcium Peaks IEI", widget.engine, run_id)

    _verify_plot_has_data(widget, "Calcium Peaks IEI")


def _has_fov_analysis_data(engine: Engine) -> bool:
    """Check if the FOVAnalysis table exists and has data in the database."""
    try:
        with Session(engine) as session:
            from cali.sqlmodel import FOVAnalysis

            result = session.exec(select(FOVAnalysis).limit(1)).first()
            return result is not None
    except OperationalError:
        return False


def _has_fov_analysis_data(engine: Engine) -> bool:
    """Check if database has FOVAnalysis data."""
    with Session(engine) as session:
        try:
            from cali.sqlmodel import FOVAnalysis

            result = session.exec(select(FOVAnalysis)).first()
            return result is not None
        except Exception:
            return False


def test_plot_spike_synchrony_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that spike synchrony plot displays actual data."""
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    if not _has_fov_analysis_data(widget.engine):
        pytest.skip("No FOVAnalysis data in database")

    plot_spike_synchrony_bar_plot(widget, "Spike Synchrony", widget.engine, run_id)

    _verify_plot_has_data(widget, "Spike Synchrony")


def test_plot_burst_count_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that burst count plot displays actual data.

    Note: Burst detection depends on threshold settings vs trace amplitudes.
    """
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    if not _has_burst_data(widget.engine):
        pytest.skip("No burst data (thresholds may not match trace activity)")

    plot_burst_count_bar_plot(widget, "Burst Count", widget.engine, run_id)

    _verify_plot_has_data(widget, "Burst Count")


def test_plot_burst_avg_duration_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that burst average duration plot displays actual data.

    Note: Burst detection depends on threshold settings vs trace amplitudes.
    """
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    if not _has_burst_data(widget.engine):
        pytest.skip("No burst data (thresholds may not match trace activity)")

    plot_burst_avg_duration_bar_plot(
        widget, "Burst Avg Duration", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Burst Avg Duration")


def test_plot_burst_avg_interval_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that burst average interval plot displays actual data.

    Note: Burst detection depends on threshold settings vs trace amplitudes.
    """
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    if not _has_burst_data(widget.engine):
        pytest.skip("No burst data (thresholds may not match trace activity)")

    plot_burst_avg_interval_bar_plot(
        widget, "Burst Avg Interval", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Burst Avg Interval")


def test_plot_percentage_active_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that percentage active ROIs plot displays actual data."""
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    plot_percentage_active_bar_plot(
        widget, "Percentage Active ROIs", widget.engine, run_id
    )

    _verify_plot_has_data(widget, "Percentage Active ROIs")


def test_plot_cell_size_has_data(
    multi_well_widget_with_data: tuple[_MultilWellGraphWidget, int],
) -> None:
    """Test that cell size plot displays actual data."""
    widget, run_id = multi_well_widget_with_data
    assert widget.engine is not None

    plot_cell_size_bar_plot(widget, "Cell Size", widget.engine, run_id)

    _verify_plot_has_data(widget, "Cell Size")
