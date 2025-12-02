"""Test GUI handling of ambiguous run scenarios."""

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication

from cali.gui._run_selection_dialog import RunSelectionDialog
from cali.sqlmodel._model import CaliResult


def test_run_selection_dialog_formatting() -> None:
    """Test that run selection dialog formats runs correctly."""
    # Create mock runs
    run1 = MagicMock(spec=CaliResult)
    run1.id = 1
    run1.detection_settings_id = 1
    run1.extraction_settings_id = 1
    run1.analysis_settings_id = None
    run1.positions_detected = [0, 1]
    run1.positions_extracted = [0, 1]
    run1.positions_analyzed = None

    run2 = MagicMock(spec=CaliResult)
    run2.id = 2
    run2.detection_settings_id = 1
    run2.extraction_settings_id = 2
    run2.analysis_settings_id = None
    run2.positions_detected = [2, 3]
    run2.positions_extracted = [2, 3]
    run2.positions_analyzed = None

    runs = [run1, run2]
    message = "Multiple runs exist with the same detection settings (ID 1)"

    # Create dialog
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = RunSelectionDialog(None, runs, message)

    # Check that runs are in the list
    assert dialog._runs_list.count() == 2

    # Check formatting
    item1_text = dialog._runs_list.item(0).text()
    assert "Run ID 1:" in item1_text
    assert "Detection: ID 1" in item1_text
    assert "Extraction: ID 1" in item1_text
    assert "Analysis: None" in item1_text
    assert "detected=[0, 1]" in item1_text

    item2_text = dialog._runs_list.item(1).text()
    assert "Run ID 2:" in item2_text
    assert "Extraction: ID 2" in item2_text


def test_run_selection_dialog_get_selected() -> None:
    """Test getting selected run ID from dialog."""
    run1 = MagicMock(spec=CaliResult)
    run1.id = 1
    run1.detection_settings_id = 1
    run1.extraction_settings_id = 1
    run1.analysis_settings_id = None
    run1.positions_detected = [0]
    run1.positions_extracted = [0]
    run1.positions_analyzed = None

    runs = [run1]
    message = "Test message"

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = RunSelectionDialog(None, runs, message)

    # Initially nothing selected
    assert dialog.get_selected_run_id() is None

    # Select first item
    dialog._runs_list.setCurrentRow(0)
    assert dialog.get_selected_run_id() == 1


@pytest.mark.skip(reason="Requires full GUI setup and interaction")
def test_gui_handles_ambiguous_detection() -> None:
    """Test that GUI properly catches and handles ambiguous detection errors.

    This is a placeholder for manual/integration testing.
    The actual test would require:
    1. Creating a test database with multiple runs
    2. Instantiating CaliGui
    3. Triggering detection-only mode
    4. Verifying dialog appears
    5. Simulating user selection
    6. Verifying correct run is used
    """
    pass
