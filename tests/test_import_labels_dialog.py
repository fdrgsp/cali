"""Tests for _ImportLabelsDialog.value() and setValue()."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cali.gui._import_labels_dialog import _ImportLabelsDialog

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot


@pytest.fixture
def dialog(populated_db: Path, qtbot: QtBot) -> _ImportLabelsDialog:
    """Open _ImportLabelsDialog backed by a real database."""
    dlg = _ImportLabelsDialog(str(populated_db))
    qtbot.addWidget(dlg)
    return dlg


def test_value_empty_initially(dialog: _ImportLabelsDialog) -> None:
    """value() returns an empty dict when no files have been assigned."""
    assert dialog.value() == {}


def test_set_value_populates_label_map(
    dialog: _ImportLabelsDialog, label_tiff: Path
) -> None:
    """setValue() with a valid fov_name -> path mapping populates _label_map."""
    fov_name = next(
        fov_name for fovs in dialog._well_fovs.values() for _, fov_name, _ in fovs
    )
    dialog.setValue({fov_name: label_tiff})

    result = dialog.value()
    assert fov_name in result
    assert result[fov_name] == label_tiff


def test_set_value_ignores_unknown_names(
    dialog: _ImportLabelsDialog, label_tiff: Path
) -> None:
    """setValue() silently ignores FOV names not present in the database."""
    dialog.setValue({"DOES_NOT_EXIST": label_tiff})
    assert dialog.value() == {}


def test_set_value_roundtrip(dialog: _ImportLabelsDialog, label_tiff: Path) -> None:
    """setValue(dialog.value()) restores the same state."""
    fov_name = next(
        fov_name for fovs in dialog._well_fovs.values() for _, fov_name, _ in fovs
    )
    dialog.setValue({fov_name: label_tiff})
    snapshot = dialog.value()

    # Clear and restore
    dialog._label_map.clear()
    dialog.setValue(snapshot)

    assert dialog.value() == snapshot


def test_on_folder_selected_populates_list(
    dialog: _ImportLabelsDialog, label_tiff: Path
) -> None:
    """Selecting a folder with TIFFs populates the available files list."""
    dialog._on_folder_selected(str(label_tiff.parent))
    assert dialog._available_list.count() >= 1
    assert len(dialog._label_files) >= 1


def test_on_folder_selected_invalid_path(dialog: _ImportLabelsDialog) -> None:
    """Non-directory path is a no-op."""
    dialog._on_folder_selected("/nonexistent/path")
    assert dialog._available_list.count() == 0


def test_auto_assign_labels_matches_fovs(
    dialog: _ImportLabelsDialog, tmp_path: Path
) -> None:
    """Auto-assign matches label files to FOVs by filename."""
    import numpy as np
    import tifffile

    # Get the actual first FOV name from the database
    fov_name = next(
        fov_name for fovs in dialog._well_fovs.values() for _, fov_name, _ in fovs
    )
    # Create a label file whose stem contains the FOV name
    label_path = tmp_path / f"{fov_name}_labels.tif"
    tifffile.imwrite(label_path, np.zeros((10, 10), dtype=np.uint16))

    dialog._label_files = [label_path]
    dialog._auto_assign_labels()

    result = dialog.value()
    assert len(result) > 0
    assert fov_name in result


def test_auto_assign_labels_no_files(dialog: _ImportLabelsDialog) -> None:
    """Auto-assign with no label files is a no-op."""
    dialog._label_files = []
    dialog._auto_assign_labels()
    assert dialog.value() == {}
