# testing data: "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
# testing database: "tests/test_data/data_and_db_for_tests/test_db.cali"
# testing database is an evoked experiment and contains 2 Runs:
#  - Run-1: without nuuropil
#  - Run-2: with nuuropil

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, select

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.plot._main_plot import SINGLE_WELL_COMBO_OPTIONS_DICT, plot_single_well_data
from cali.sqlmodel import FOV, CaliResult

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine

# Test data paths
TEST_DB = Path(__file__).parent / "test_data" / "data_and_db_for_tests" / "test_db.cali"


@pytest.fixture
def db_engine() -> Generator[Engine, None, None]:
    """Create database engine for test database."""
    db_path = TEST_DB
    assert db_path.exists(), f"Test database not found: {db_path}"
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

    yield engine
    engine.dispose()  # Close all connections


@pytest.fixture
def fov_name(db_engine: Engine) -> str:
    """Get the first FOV name from the database."""
    with Session(db_engine) as session:
        fov = session.exec(select(FOV)).first()
        assert fov is not None, "No FOV found in database"
        return fov.name


@pytest.fixture
def run_ids(db_engine: Engine) -> list[int]:
    """Get all run IDs from the database."""
    with Session(db_engine) as session:
        results = session.exec(select(CaliResult.id)).all()
        assert len(results) >= 1, "No runs found in database"
        return [r for r in results if r is not None]


@pytest.fixture
def widget(qtbot: QtBot) -> Generator[_SingleWellGraphWidget, None, None]:
    """Create a _SingleWellGraphWidget for testing."""
    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)
    yield widget
    # Clean up engine to prevent ResourceWarning
    if widget.engine is not None:
        widget.engine.dispose(close=True)
        widget.engine = None


def get_all_plot_names() -> list[str]:
    """Get all available plot names from the registry."""
    plots = []
    for _category, names in SINGLE_WELL_COMBO_OPTIONS_DICT.items():
        plots.extend(names)
    return plots


@pytest.mark.parametrize("plot_name", get_all_plot_names())
def test_all_plots_render_without_error(
    plot_name: str,
    widget: _SingleWellGraphWidget,
    db_engine: Engine,
    fov_name: str,
    run_ids: list[int],
) -> None:
    """Test that all plots can be rendered without errors."""
    widget.engine = db_engine

    # Use the first run ID
    run_id = run_ids[0]

    # Attempt to plot - should not raise any exceptions
    try:
        plot_single_well_data(
            widget=widget,
            engine=db_engine,
            fov_name=fov_name,
            text=plot_name,
            run_id=run_id,
            rois=None,  # Test with all ROIs
        )

        # Verify plot has been populated
        assert widget.plot_item is not None, f"Plot item not created for {plot_name}"

        # Some plots might not have items if there's insufficient data, that's ok
        # The important thing is no exceptions were raised

    except Exception as e:
        pytest.fail(f"Plot '{plot_name}' raised exception: {e}")


@pytest.mark.parametrize("plot_name", get_all_plot_names())
def test_plots_with_roi_subset(
    plot_name: str,
    widget: _SingleWellGraphWidget,
    db_engine: Engine,
    fov_name: str,
    run_ids: list[int],
) -> None:
    """Test plots with a subset of ROIs selected."""
    widget.engine = db_engine
    run_id = run_ids[0]

    # Test with first 5 ROIs
    roi_subset = list(range(1, 6))

    try:
        plot_single_well_data(
            widget=widget,
            engine=db_engine,
            fov_name=fov_name,
            text=plot_name,
            run_id=run_id,
            rois=roi_subset,
        )

        assert widget.plot_item is not None, f"Plot item not created for {plot_name}"

    except Exception as e:
        pytest.fail(f"Plot '{plot_name}' with ROI subset raised exception: {e}")


def test_multiple_runs(
    widget: _SingleWellGraphWidget, db_engine: Engine, fov_name: str, run_ids: list[int]
) -> None:
    """Test that plots work with different runs."""
    # Test a representative plot with each run
    plot_name = "Calcium Deconvolved ΔF/F0 Traces with Peaks"

    for run_id in run_ids:
        plot_single_well_data(
            widget=widget,
            engine=db_engine,
            fov_name=fov_name,
            text=plot_name,
            run_id=run_id,
            rois=None,
        )

        assert widget.plot_item is not None
        assert len(widget.plot_item.items) > 0, f"No items for run {run_id}"  # type: ignore[union-attr]


def test_widget_clear_plot(
    widget: _SingleWellGraphWidget, db_engine: Engine, fov_name: str, run_ids: list[int]
) -> None:
    """Test that clear_plot properly resets the widget."""
    widget.engine = db_engine
    run_id = run_ids[0]

    # Plot something
    plot_single_well_data(
        widget=widget,
        engine=db_engine,
        fov_name=fov_name,
        text="Calcium ΔF/F0 Traces",
        run_id=run_id,
        rois=None,
    )

    assert len(widget.plot_item.items) > 0  # type: ignore[union-attr]

    # Clear the plot
    widget.clear_plot()

    # Check that plot is cleared
    assert len(widget.plot_item.items) == 0  # type: ignore[union-attr]
    assert widget.plot_item.titleLabel.text == ""  # type: ignore[union-attr]

    # Check colorbar is removed
    assert widget.colorbar is None


def test_plot_with_single_roi(
    widget: _SingleWellGraphWidget, db_engine: Engine, fov_name: str, run_ids: list[int]
) -> None:
    """Test plots with single ROI selected (should use white color)."""
    widget.engine = db_engine
    run_id = run_ids[0]

    # Test with single ROI
    plot_single_well_data(
        widget=widget,
        engine=db_engine,
        fov_name=fov_name,
        text="Calcium ΔF/F0 Traces",
        run_id=run_id,
        rois=[1],  # Single ROI
    )

    assert widget.plot_item is not None
    assert len(widget.plot_item.items) > 0  # type: ignore[union-attr]
