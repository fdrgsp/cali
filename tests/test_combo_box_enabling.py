"""Test GUI plot widget behavior: combo boxes, FOV changes, and plot display.

This module consolidates tests for:
- Combo box enabling/disabling based on pipeline stage availability
- Plot display and rendering
- FOV switching behavior
- Combo selection persistence across state changes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import Qt
from sqlmodel import Session, create_engine, select

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.sqlmodel._model import FOV

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot


@pytest.fixture(scope="function")
def widget_with_db(
    qtbot: QtBot,
) -> Generator[tuple[_SingleWellGraphWidget, str, str], None, None]:
    """Create a widget connected to the test database with full pipeline results."""
    # Use existing test database
    db_path = "tests/test_data/multi_pos/result_2pos.cali"

    # Get the FOV name from the database
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        fov_name = session.exec(select(FOV.name).limit(1)).first()

    assert fov_name is not None

    # Create fresh widget for each test
    widget = _SingleWellGraphWidget(None)  # type: ignore[arg-type]
    qtbot.addWidget(widget)

    # Connect to database but don't set FOV or run_id
    widget.database_path = db_path
    widget.engine = engine

    # Explicitly ensure clean state
    widget._fov = ""
    widget._run_id = None

    yield widget, db_path, fov_name

    engine.dispose(close=True)


# ============================================================================
# Combo Box Enabling/Disabling Tests
# ============================================================================


def test_combo_disabled_without_fov_or_run(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo items are disabled when no FOV or run_id is set."""
    widget, _, _ = widget_with_db

    # Initial state - no FOV or run_id
    has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
    assert not has_det
    assert not has_ext
    assert not has_ana

    # Count disabled items
    # (38 plots for spontaneous experiment - evoked plots not shown yet)
    # (Added thresholded spike intensity heatmap)
    model = widget._combo.model()
    disabled_count = sum(
        1
        for i in range(model.rowCount())
        if not (model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)  # Skip sections
        and model.item(i).text() != "None"
    )

    # All plots should be disabled (38 plots require pipeline stages)
    assert disabled_count == 38


def test_combo_disabled_with_only_run_id(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo items remain disabled when only run_id is set."""
    widget, _, _ = widget_with_db

    # Set run_id but not FOV
    widget.run_id = 1

    has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
    assert not has_det  # No FOV, so no detection
    assert not has_ext
    assert not has_ana

    # All items should still be disabled
    # (61 plots: 47 spontaneous + 14 evoked, because exp_type is "Evoked Activity")
    # (Added thresholded spike intensity heatmap + 6 new stim/non-stim heatmaps)
    model = widget._combo.model()
    disabled_count = sum(
        1
        for i in range(model.rowCount())
        if not (model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)
        and model.item(i).text() != "None"
    )

    assert disabled_count == 61


def test_combo_enabled_with_fov_and_run_id(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo items are enabled when both FOV and run_id are set."""
    widget, _, fov_name = widget_with_db

    # Set run_id and FOV
    widget.run_id = 1
    widget.fov = fov_name

    # All pipeline stages should be available
    has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
    assert has_det
    assert has_ext
    assert has_ana

    # All items should be enabled
    model = widget._combo.model()
    enabled_count = sum(
        1
        for i in range(model.rowCount())  # type: ignore[union-attr]
        if (model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)  # type: ignore[union-attr]
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)  # type: ignore[union-attr]
        and model.item(i).text() != "None"  # type: ignore[union-attr]
    )

    # Spontaneous plots enabled (not evoked experiment)
    assert enabled_count > 0  # At least some plots should be enabled


def test_combo_updates_on_fov_change(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo box updates when FOV changes."""
    widget, _, fov_name = widget_with_db

    # Set run_id first
    widget.run_id = 1

    # Initially disabled
    has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
    assert not has_det

    # Now set FOV - this should trigger _update_combo_box()
    widget.fov = fov_name

    # Now should be enabled
    has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
    assert has_det
    assert has_ext
    assert has_ana
