"""Tests for save-as dialog widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from cali.gui._save_as_widgets import _SaveLabelsAsTiff

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_save_labels_as_tiff_default_values(qtbot: QtBot) -> None:
    """Test _SaveLabelsAsTiff default state."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    # Default: empty positions, "All" detection, no overwrite
    path, positions, det_id, overwrite = dialog.value()
    assert path == ""
    assert positions == []
    assert det_id is None
    assert overwrite is False


def test_save_labels_as_tiff_populate_detection_settings(qtbot: QtBot) -> None:
    """Test populate_detection_settings fills the combo correctly."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    settings = [(1, "suite2p"), (3, "cellpose")]
    dialog.populate_detection_settings(settings)

    combo = dialog._detection_combo
    assert combo.count() == 3  # "All" + 2 settings
    assert combo.itemText(0) == "All"
    assert combo.itemData(0) is None
    assert combo.itemText(1) == "Detection ID 1 (suite2p)"
    assert combo.itemData(1) == 1
    assert combo.itemText(2) == "Detection ID 3 (cellpose)"
    assert combo.itemData(2) == 3


def test_save_labels_as_tiff_auto_select_single(qtbot: QtBot) -> None:
    """Test that a single detection setting is auto-selected."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    dialog.populate_detection_settings([(5, "cellpose")])

    assert dialog._detection_combo.currentIndex() == 1
    assert dialog._detection_combo.currentData() == 5


def test_save_labels_as_tiff_positions_parsing(qtbot: QtBot) -> None:
    """Test position parsing from the line edit."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    dialog._pos_line_edit.setText("0-2, 5")
    _, positions, _, _ = dialog.value()
    assert positions == [0, 1, 2, 5]


def test_save_labels_as_tiff_overwrite_checkbox(qtbot: QtBot) -> None:
    """Test overwrite checkbox value."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    dialog._overwrite_cb.setChecked(True)
    _, _, _, overwrite = dialog.value()
    assert overwrite is True


def test_save_labels_as_tiff_accept_requires_valid_path(
    qtbot: QtBot, tmp_path: object
) -> None:
    """Test that accept rejects empty/invalid paths."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    with patch("cali.gui._save_as_widgets.show_error_dialog"):
        # Empty path should not accept
        dialog._browse_widget.setValue("")
        dialog.accept()
        assert not dialog.result()  # dialog should not be accepted

        # Non-existent path should not accept
        dialog._browse_widget.setValue("/nonexistent/path")
        dialog.accept()
        assert not dialog.result()


def test_save_labels_as_tiff_accept_valid_path(qtbot: QtBot, tmp_path: object) -> None:
    """Test that accept works with a valid directory."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    dialog._browse_widget.setValue(str(tmp_path))
    dialog.accept()
    assert dialog.result()  # dialog should be accepted


def test_save_labels_as_tiff_detection_selection(qtbot: QtBot) -> None:
    """Test selecting a specific detection setting returns correct ID."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    dialog.populate_detection_settings([(1, "suite2p"), (2, "cellpose")])
    dialog._detection_combo.setCurrentIndex(2)  # select "Detection ID 2 (cellpose)"

    _, _, det_id, _ = dialog.value()
    assert det_id == 2


def test_save_labels_as_tiff_repopulate_clears(qtbot: QtBot) -> None:
    """Test that calling populate_detection_settings again clears old items."""
    dialog = _SaveLabelsAsTiff()
    qtbot.addWidget(dialog)

    dialog.populate_detection_settings([(1, "suite2p"), (2, "cellpose")])
    assert dialog._detection_combo.count() == 3

    dialog.populate_detection_settings([(10, "stardist")])
    assert dialog._detection_combo.count() == 2  # "All" + 1
    assert dialog._detection_combo.currentData() == 10  # auto-selected
