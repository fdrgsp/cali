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
    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"

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


def _count_combo_items(widget: _SingleWellGraphWidget, *, enabled: bool) -> int:
    """Count enabled or disabled combo box items (excluding sections and 'None')."""
    model = widget._combo.model()
    return sum(
        1
        for i in range(model.rowCount())
        if (bool(model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled) == enabled)
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)  # Skip sections
        and model.item(i).text() != "None"
    )


def _assert_pipeline_stages(
    widget: _SingleWellGraphWidget, *, has_det: bool, has_ext: bool, has_ana: bool
) -> None:
    """Assert pipeline stage availability matches expectations."""
    actual_det, actual_ext, actual_ana = widget._check_pipeline_stage_availability()
    assert actual_det == has_det
    assert actual_ext == has_ext
    assert actual_ana == has_ana


# ============================================================================
# Combo Box Enabling/Disabling Tests
# ============================================================================


def test_combo_disabled_without_fov_or_run(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo items are disabled when no FOV or run_id is set."""
    widget, _, _ = widget_with_db

    # Initial state - no FOV or run_id
    _assert_pipeline_stages(widget, has_det=False, has_ext=False, has_ana=False)

    # All plots should be disabled (removed spike correlation plot)
    assert _count_combo_items(widget, enabled=False) == 43


def test_combo_disabled_with_only_run_id(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo items remain disabled when only run_id is set."""
    widget, _, _ = widget_with_db

    # Set run_id but not FOV
    widget.run_id = 1

    # No FOV, so no pipeline stages available
    _assert_pipeline_stages(widget, has_det=False, has_ext=False, has_ana=False)

    # All items should still be disabled
    # (57 plots: spontaneous + evoked)
    assert _count_combo_items(widget, enabled=False) == 57


def test_combo_enabled_with_fov_and_run_id(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo items are enabled when both FOV and run_id are set."""
    widget, _, fov_name = widget_with_db

    # Set run_id and FOV
    widget.run_id = 1
    widget.fov = fov_name

    # All pipeline stages should be available
    _assert_pipeline_stages(widget, has_det=True, has_ext=True, has_ana=True)

    # At least some plots should be enabled
    assert _count_combo_items(widget, enabled=True) > 0


def test_combo_updates_on_fov_change(
    widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Test that combo box updates when FOV changes."""
    widget, _, fov_name = widget_with_db

    # Set run_id first
    widget.run_id = 1

    # Initially no pipeline stages (no FOV yet)
    _assert_pipeline_stages(widget, has_det=False, has_ext=False, has_ana=False)

    # Now set FOV - this should trigger _update_combo_box()
    widget.fov = fov_name

    # Now all pipeline stages should be available
    _assert_pipeline_stages(widget, has_det=True, has_ext=True, has_ana=True)
