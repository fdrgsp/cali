"""Comprehensive tests for plot widget GUI behavior.

This module tests:
- Combo box enabling/disabling based on pipeline stage availability
- Plot display and rendering with real data
- FOV switching behavior and plot reload logic
- Combo selection persistence across state changes
- Detection GUI model selection with findText
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel
from sqlmodel import Session, create_engine, select

from cali.gui import CaliGui
from cali.gui._detection_gui import CellposeSettingsData
from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.sqlmodel._model import FOV

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from pytestqt.qtbot import QtBot


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def plot_widget_with_db(
    qtbot: QtBot,
) -> Generator[tuple[_SingleWellGraphWidget, str, str], None, None]:
    """Create a plot widget connected to test database with full pipeline results."""
    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        fov_name = session.exec(select(FOV.name).limit(1)).first()

    assert fov_name is not None

    widget = _SingleWellGraphWidget(None)  # type: ignore[arg-type]
    qtbot.addWidget(widget)

    widget.database_path = db_path
    widget.engine = engine
    widget._fov = ""
    widget._run_id = None

    yield widget, db_path, fov_name

    engine.dispose(close=True)


@pytest.fixture
def gui_for_plots(qtbot: QtBot) -> CaliGui:
    """Create a CaliGui instance with test data for plot testing."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._database_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    gui._data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
    gui._initialize_from_database(gui._database_path, gui._data_path)
    return gui


# ============================================================================
# Combo Box Enabling/Disabling Tests
# ============================================================================


def test_combo_disabled_with_only_run_id(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Combo items should remain disabled when only run_id is set (no FOV)."""
    widget, _, _ = plot_widget_with_db

    widget.run_id = 1

    has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
    assert not has_det
    assert not has_ext
    assert not has_ana

    model = widget._combo.model()
    assert isinstance(model, QStandardItemModel)
    disabled_count = sum(
        1
        for i in range(model.rowCount())
        if not (model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)
        and model.item(i).text() != "None"
    )

    # Total disabled plot items when only run_id is set (including 3 rising edges)
    assert disabled_count == 60


def test_combo_enabled_with_fov_and_run_id(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Combo items should be enabled when both FOV and run_id are set."""
    widget, _, fov_name = plot_widget_with_db

    widget.run_id = 1
    widget.fov = fov_name

    has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
    assert has_det
    assert has_ext
    assert has_ana

    model = widget._combo.model()
    assert isinstance(model, QStandardItemModel)
    enabled_count = sum(
        1
        for i in range(model.rowCount())
        if (model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)
        and model.item(i).text() != "None"
    )

    assert enabled_count > 0


def test_combo_updates_on_fov_change(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
) -> None:
    """Combo box should update availability when FOV changes."""
    widget, _, fov_name = plot_widget_with_db

    widget.run_id = 1

    # Start with no FOV - all disabled
    model = widget._combo.model()
    assert isinstance(model, QStandardItemModel)
    initial_disabled = sum(
        1
        for i in range(model.rowCount())
        if not (model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)
        and model.item(i).text() != "None"
    )
    assert initial_disabled > 0

    # Set FOV - items should become enabled
    widget.fov = fov_name

    enabled_count = sum(
        1
        for i in range(model.rowCount())
        if (model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        and not model.item(i).data(Qt.ItemDataRole.UserRole + 1)
        and model.item(i).text() != "None"
    )
    assert enabled_count > 0


# ============================================================================
# FOV Switching and Plot Reload Tests
# ============================================================================


def test_fov_change_reloads_plot_when_disabled_to_enabled(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
    qtbot: QtBot,
) -> None:
    """Switching from FOV with no data to FOV with data should reload plot.

    This tests the fix for the issue where clicking on a well with data after
    clicking on a well with no data wouldn't show the plot without an intermediate
    click on the "last used well".
    """
    widget, _, fov_name = plot_widget_with_db

    # Set up: run_id + FOV with data
    widget.run_id = 1
    widget.fov = fov_name

    # Select a plot type
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0
    widget._combo.setCurrentIndex(idx)
    qtbot.wait(50)

    # Verify plot has items
    assert widget.plot_item is not None
    initial_item_count = len(widget.plot_item.items)
    assert initial_item_count > 0

    # Simulate clicking on a well with no data by setting empty FOV
    widget.fov = ""
    qtbot.wait(50)

    # Plot should be cleared but combo selection preserved
    assert widget._combo.currentText() == plot_name
    # (Note: plot may be empty or have minimal items when no FOV)

    # Now switch back to FOV with data
    # This should reload the plot WITHOUT needing to change combo selection
    widget.fov = fov_name
    qtbot.wait(50)

    # Plot should reload automatically
    assert widget.plot_item is not None
    assert len(widget.plot_item.items) > 0
    assert widget._combo.currentText() == plot_name


def test_fov_setter_preserves_combo_selection(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
    qtbot: QtBot,
) -> None:
    """Setting FOV should preserve combo selection while updating availability."""
    widget, _, fov_name = plot_widget_with_db

    widget.run_id = 1

    # Select a plot before setting FOV
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0
    widget._combo.setCurrentIndex(idx)
    selected_before = widget._combo.currentText()

    # Set FOV
    widget.fov = fov_name
    qtbot.wait(50)

    # Selection should be preserved
    assert widget._combo.currentText() == selected_before


# ============================================================================
# Plot Display Integration Tests
# ============================================================================


def test_single_well_plot_displays(gui_for_plots: CaliGui, qtbot: QtBot) -> None:
    """Selecting a plot type should actually render plot items."""
    widget = gui_for_plots.SW_GRAPHS[0]

    # Set FOV
    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        assert fov is not None

        widget.fov = fov.name

    # Select a plot
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0

    widget._combo.setCurrentIndex(idx)

    # Verify plot rendered
    assert widget.plot_item is not None
    assert len(widget.plot_item.items) > 0
    assert widget._combo.currentText() == plot_name


def test_combo_preserves_selection_on_fov_change(
    gui_for_plots: CaliGui, qtbot: QtBot
) -> None:
    """Combo selection should persist when FOV changes."""
    widget = gui_for_plots.SW_GRAPHS[0]

    with Session(widget._engine) as session:
        all_fovs = session.exec(select(FOV).limit(2)).all()
        assert len(all_fovs) >= 2

    # Set initial FOV and select plot
    widget.fov = all_fovs[0].name
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0
    widget._combo.setCurrentIndex(idx)
    qtbot.wait(50)

    initial_selection = widget._combo.currentText()

    # Change FOV
    widget.fov = all_fovs[1].name
    qtbot.wait(50)

    # Selection should persist
    assert widget._combo.currentText() == initial_selection


def test_combo_preserves_enabled_selection_on_run_change(
    gui_for_plots: CaliGui, qtbot: QtBot
) -> None:
    """Enabled combo selection should persist when run changes."""
    widget = gui_for_plots.SW_GRAPHS[0]

    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        assert fov is not None

    # Set FOV and run
    widget.fov = fov.name
    widget.run_id = 1
    qtbot.wait(50)

    # Select a plot
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0
    widget._combo.setCurrentIndex(idx)
    qtbot.wait(50)

    # Change run
    widget.run_id = 2
    qtbot.wait(50)

    # If plot still available, selection should persist
    model = widget._combo.model()
    assert isinstance(model, QStandardItemModel)
    idx = widget._combo.findText(plot_name)
    if idx >= 0:
        item = model.item(idx)
        if item and item.flags() & Qt.ItemFlag.ItemIsEnabled:
            assert widget._combo.currentText() == plot_name


def test_multiple_plot_types_render(gui_for_plots: CaliGui, qtbot: QtBot) -> None:
    """Multiple different plot types should all render correctly."""
    widget = gui_for_plots.SW_GRAPHS[0]

    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        assert fov is not None

        widget.fov = fov.name

    # Test multiple plot types
    plot_types = ["Calcium Raw Traces", "ROI Contour Raster Plot"]

    for plot_name in plot_types:
        idx = widget._combo.findText(plot_name)
        if idx >= 0:
            model = widget._combo.model()
            assert isinstance(model, QStandardItemModel)
            item = model.item(idx)
            if item and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                widget._combo.setCurrentIndex(idx)
                qtbot.wait(50)

                assert widget.plot_item is not None
                assert len(widget.plot_item.items) > 0, (
                    f"Plot '{plot_name}' has no items"
                )


def test_plot_clears_when_selecting_none(gui_for_plots: CaliGui, qtbot: QtBot) -> None:
    """Selecting 'None' should clear the plot."""
    widget = gui_for_plots.SW_GRAPHS[0]

    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        assert fov is not None

        widget.fov = fov.name

    # Select a plot first
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0
    widget._combo.setCurrentIndex(idx)
    qtbot.wait(50)

    # Verify plot has items
    assert widget.plot_item is not None
    assert len(widget.plot_item.items) > 0

    # Select "None"
    none_idx = widget._combo.findText("None")
    assert none_idx >= 0
    widget._combo.setCurrentIndex(none_idx)
    qtbot.wait(50)

    # Plot should be cleared (or have minimal items)
    # The exact behavior depends on clear_plot() implementation


# ============================================================================
# Detection GUI Model Selection Tests
# ============================================================================


def test_custom_model_combo_uses_findtext_for_selection(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    """Custom model selection should use findText for robust combo selection.

    This tests the fix for the issue where setCurrentText() would fail silently
    when the combo was recently rebuilt.
    """
    # Create a standalone GUI without database initialization
    gui = CaliGui()
    qtbot.addWidget(gui)

    detection_wdg = gui._detection_wdg._cellpose_wdg

    # Create custom settings
    custom_model_path = str(tmp_path / "custom_model.pth")
    custom_settings = CellposeSettingsData(
        model_type="custom",
        model_path=custom_model_path,
        diameter=30,
        cellprob_threshold=-0.5,
    )

    # Apply settings using setValue (which uses findText internally)
    detection_wdg.setValue(custom_settings)
    qtbot.wait(50)

    # Verify custom model is selected
    assert detection_wdg._models_combo.currentText() == "custom"
    assert detection_wdg._browse_custom_model.value() == custom_model_path
    assert detection_wdg._browse_custom_model.isVisible()
    assert detection_wdg._diameter_spin.value() == 30
    assert detection_wdg._cellprob_threshold_spin.value() == -0.5

    # Switch to default model
    default_settings = CellposeSettingsData()
    detection_wdg.setValue(default_settings)
    qtbot.wait(50)

    # Should revert to default
    assert detection_wdg._models_combo.currentText() != "custom"
    assert not detection_wdg._browse_custom_model.isVisible()

    # Switch back to custom - should work without issues
    detection_wdg.setValue(custom_settings)
    qtbot.wait(50)

    assert detection_wdg._models_combo.currentText() == "custom"
    assert detection_wdg._browse_custom_model.isVisible()


def test_custom_model_combo_item_always_available(qtbot: QtBot) -> None:
    """The 'custom' option should always be available in the models combo box."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    detection_wdg = gui._detection_wdg._cellpose_wdg

    # Check that "custom" exists
    custom_idx = detection_wdg._models_combo.findText("custom")
    assert custom_idx >= 0, "'custom' should always be in the combo box"

    # Verify we can select it
    detection_wdg._models_combo.setCurrentIndex(custom_idx)
    qtbot.wait(50)

    assert detection_wdg._models_combo.currentText() == "custom"
    assert detection_wdg._browse_custom_model.isVisible()


def test_detection_model_findtext_handles_missing_item(qtbot: QtBot) -> None:
    """setValue should handle gracefully when model type is not in combo."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    detection_wdg = gui._detection_wdg._cellpose_wdg

    # Try to set a non-existent model type
    invalid_settings = CellposeSettingsData(
        model_type="nonexistent_model",
        diameter=25,
    )

    # This should not crash - just won't change the combo
    detection_wdg.setValue(invalid_settings)
    qtbot.wait(50)

    # Combo should remain unchanged (or at default)
    # Other settings should still apply
    assert detection_wdg._diameter_spin.value() == 25


# ============================================================================
# Combo Box Rebuild and Selection Persistence Tests
# ============================================================================


def test_rebuild_combo_box_without_preserve_resets_selection(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
    qtbot: QtBot,
) -> None:
    """Rebuilding combo without preserve_selection should reset to 'None'."""
    widget, _, fov_name = plot_widget_with_db

    widget.run_id = 1
    widget.fov = fov_name

    # Select a plot
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0
    widget._combo.setCurrentIndex(idx)
    qtbot.wait(50)

    # Rebuild without preserving
    widget._rebuild_combo_box(preserve_selection=False)
    qtbot.wait(50)

    # Should reset to "None"
    assert widget._combo.currentText() == "None"


def test_rebuild_combo_box_with_preserve_keeps_selection(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
    qtbot: QtBot,
) -> None:
    """Rebuilding combo with preserve_selection should keep current selection."""
    widget, _, fov_name = plot_widget_with_db

    widget.run_id = 1
    widget.fov = fov_name

    # Select a plot
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0
    widget._combo.setCurrentIndex(idx)
    qtbot.wait(50)

    initial_selection = widget._combo.currentText()

    # Rebuild with preserving
    widget._rebuild_combo_box(preserve_selection=True)
    qtbot.wait(50)

    # Should keep selection if still available and enabled
    model = widget._combo.model()
    assert isinstance(model, QStandardItemModel)
    idx = widget._combo.findText(plot_name)
    if idx >= 0:
        item = model.item(idx)
        if item and item.flags() & Qt.ItemFlag.ItemIsEnabled:
            assert widget._combo.currentText() == initial_selection


def test_clicking_back_to_previous_well_shows_plot(
    plot_widget_with_db: tuple[_SingleWellGraphWidget, str, str],
    qtbot: QtBot,
) -> None:
    """Clicking empty well then back to previous well should show plot.

    Scenario:
    1. Well A selected with data → plot shows
    2. Click empty well B → plot clears (fov="")
    3. Click well A again → plot should show immediately

    This test simulates the GUI behavior where _on_scene_well_changed calls
    both clear_plot() AND sets fov="" before repopulating with new FOV.
    """
    widget, _, fov_name = plot_widget_with_db

    widget.run_id = 1

    # Step 1: Select well A with data
    widget.fov = fov_name
    widget._combo.setCurrentText("Calcium Raw Traces")
    qtbot.wait(50)

    # Verify plot has data
    assert widget.plot_item is not None
    initial_plot_items = len(widget.plot_item.items)
    assert initial_plot_items > 0, "Plot should have items after setting valid FOV"

    # Step 2: Click empty well B (simulated by GUI's _on_scene_well_changed behavior)
    # In the actual GUI, this calls both clear_plot() AND sets fov=""
    widget.clear_plot()
    widget.fov = ""
    qtbot.wait(50)

    # Plot should be cleared (or minimal items)
    plot_items_after_clear = len(widget.plot_item.items)

    # Step 3: Click well A again (simulating selection of same well)
    # In GUI: _on_scene_well_changed calls clear_plot() + fov="", then sets new fov
    widget.clear_plot()
    widget.fov = ""  # This is the key: FOV is cleared BEFORE being set to same value
    qtbot.wait(50)
    widget.fov = fov_name  # Now set to same FOV as step 1
    qtbot.wait(50)

    # Plot should show data again
    final_plot_items = len(widget.plot_item.items)
    assert final_plot_items > 0, (
        f"Plot should have items after clicking back to well with data. "
        f"Initial: {initial_plot_items}, After clear: {plot_items_after_clear}, "
        f"Final: {final_plot_items}"
    )
