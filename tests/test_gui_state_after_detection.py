"""Test GUI state after detection/analysis runs.

This test verifies that GUI components are properly enabled after running
detection and analysis, which was a bug introduced in commit d4742d6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import Qt

from cali.gui import CaliGui

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def gui_with_detection(qtbot: QtBot) -> CaliGui:
    """Create a GUI loaded with a database that has detection results."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Load the test database that has detection results
    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"

    gui._initialize_from_database(db_path, data_path)
    qtbot.waitUntil(lambda: gui._loading_bar.isHidden(), timeout=10000)

    return gui


def test_label_button_enabled_after_loading_database(
    gui_with_detection: CaliGui, qtbot: QtBot
) -> None:
    """Test label button is enabled after loading database with detection.

    This test simulates the ACTUAL user workflow:
    1. Load database with detection results
    2. Click on a well
    3. Select FOV from table
    4. Verify label button is enabled

    This is a regression test for the bug where labels would not appear.
    """
    # The GUI fixture already loaded the database
    # Now we need to simulate selecting a well and FOV

    # First, let's manually trigger what happens when you click on B5
    assert gui_with_detection._data is not None
    assert gui_with_detection._data.sequence is not None

    # Get B5_0000 position (should be position 0)
    positions = gui_with_detection._data.sequence.stage_positions

    b5_pos = None
    for i, pos in enumerate(positions):
        # Look for the FULL FOV name with position index
        if hasattr(pos, "name") and pos.name and "B5_0000" in pos.name:
            b5_pos = (i, pos)
            break

    if b5_pos is None:
        pytest.skip("B5_0000 position not found in test data")

    pos_idx, pos = b5_pos

    # Simulate well selection by calling the handler directly
    # This is what happens when you click on a well
    from cali.gui._fov_table import WellInfo

    well_info = WellInfo(pos_idx=pos_idx, fov=pos)

    # Add the position to the FOV table (simulating what happens on well click)
    gui_with_detection._fov_table.clear()
    gui_with_detection._fov_table.setRowCount(0)
    gui_with_detection._fov_table.add_position(well_info)

    qtbot.wait(100)

    # Now select the FOV in the table (simulating user click)
    gui_with_detection._fov_table.selectRow(0)
    qtbot.wait(200)

    # Verify labels were loaded
    roi_labels, neuropil_labels = gui_with_detection._get_labels(well_info)
    assert roi_labels is not None, "ROI labels should exist for B5_0000"
    assert neuropil_labels is not None, "Neuropil labels should exist for B5_0000"

    # Verify the label button is enabled
    assert gui_with_detection._image_viewer._labels.isEnabled(), (
        "Label button should be enabled when detection results exist for the FOV"
    )


def test_label_button_enabled_after_detection(
    gui_with_detection: CaliGui, qtbot: QtBot
) -> None:
    """Test label button in image viewer is enabled when setData gets labels.

    This test verifies the fix for the bug where the label button would never
    be enabled even when labels existed. The bug was that setData() was
    checking for labels_image and contours_image existence BEFORE
    update_image() created them.

    This is a regression test for the bug introduced in commit d4742d6.
    """
    import numpy as np

    # Create dummy image and label data (simulating what would come from DB)
    image_data = np.random.rand(512, 512).astype(np.float32)

    # Create a simple label mask with a few ROIs
    labels = np.zeros((512, 512), dtype=np.uint16)
    labels[50:100, 50:100] = 1  # ROI 1
    labels[150:200, 150:200] = 2  # ROI 2
    labels[250:300, 250:300] = 3  # ROI 3

    # Call setData with both image and labels (this is what
    # _on_fov_table_selection_changed does)
    gui_with_detection._image_viewer.setData(image_data, labels)

    qtbot.wait(100)

    # Verify the label button is enabled
    assert gui_with_detection._image_viewer._labels.isEnabled(), (
        "Label button should be enabled when setData is called with labels"
    )

    # Verify images were created
    assert gui_with_detection._image_viewer._viewer.image is not None
    assert gui_with_detection._image_viewer._viewer.labels_image is not None
    assert gui_with_detection._image_viewer._viewer.contours_image is not None

    # Verify the button tooltip is correct (should be the enabled tooltip)
    tooltip = gui_with_detection._image_viewer._labels.toolTip()
    assert "Toggle ROI labels visibility" in tooltip, (
        f"Expected enabled tooltip, got: {tooltip}"
    )


def test_visualization_combo_enabled_after_analysis(
    gui_with_detection: CaliGui, qtbot: QtBot
) -> None:
    """Test visualization combos have enabled items after loading analysis.

    This test verifies that combo boxes in the visualization tab are properly
    populated and have enabled items when analysis results exist.
    """
    # Switch to visualization tab
    gui_with_detection._main_tab.setCurrentWidget(gui_with_detection._visualization_tab)
    qtbot.wait(100)

    # Select a well and FOV (same as above)
    assert gui_with_detection._data is not None
    assert gui_with_detection._data.sequence is not None

    pos = gui_with_detection._data.sequence.stage_positions[0]
    scene = gui_with_detection._plate_view.scene()
    items = scene.items()  # type: ignore

    for item in items:
        if hasattr(item, "well_pos") and item.well_pos == pos:  # type: ignore
            gui_with_detection._plate_view.clearSelection()
            item.setSelected(True)
            break

    qtbot.wait(100)

    if gui_with_detection._fov_table.rowCount() > 0:
        gui_with_detection._fov_table.selectRow(0)
        qtbot.wait(100)

    # Check that at least one graph widget has enabled combo items
    graph_widget = gui_with_detection._single_well_graph_1

    # Verify combo box is enabled
    assert graph_widget._combo.isEnabled(), (
        "Plot selection combo should be enabled when analysis results exist"
    )

    # Verify at least some items are enabled
    model = graph_widget._combo.model()
    enabled_items = []
    for i in range(model.rowCount()):  # type: ignore
        item = model.item(i)  # type: ignore
        if item and (item.flags() & Qt.ItemFlag.ItemIsEnabled):  # type: ignore
            enabled_items.append(item.text())

    assert len(enabled_items) > 0, (
        f"At least some plot options should be enabled. "
        f"Total items: {model.rowCount()}, Enabled: {len(enabled_items)}"  # type: ignore
    )


def test_labels_button_disabled_when_no_labels(qtbot: QtBot) -> None:
    """Test that label button is disabled when no labels exist.

    This is a control test to verify the button is properly disabled
    when there are no detection results.
    """
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Don't load any database - just verify initial state
    assert not gui._image_viewer._labels.isEnabled(), (
        "Label button should be disabled initially when no labels exist"
    )

    # Set data without labels
    import numpy as np

    data = np.random.rand(100, 100).astype(np.float32)
    gui._image_viewer.setData(data, labels=None)

    # Verify button is still disabled
    assert not gui._image_viewer._labels.isEnabled(), (
        "Label button should remain disabled when setData is called with labels=None"
    )

    # Verify tooltip indicates why it's disabled
    tooltip = gui._image_viewer._labels.toolTip()
    assert "No labels data available" in tooltip, (
        f"Expected disabled tooltip explaining why, got: {tooltip}"
    )
