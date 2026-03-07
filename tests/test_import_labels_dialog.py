"""Tests for _ImportLabelsDialog.value() and setValue()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import tifffile

from cali.gui._import_labels_dialog import _ImportLabelsDialog
from cali.runner import CaliRunner
from cali.sqlmodel._model import DetectionSettings

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from pytestqt.qtbot import QtBot


@pytest.fixture
def populated_db(
    tmp_path: Path,
    test_experiment: Any,
    data_path: Path,
    mock_detection_runner: MagicMock,
) -> Path:
    """Database with one detected FOV (FOV name: A1_0000)."""
    db_path = tmp_path / "dialog_test.cali"
    runner = CaliRunner(commit_batch_size=1)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
        database_name=db_path.name,
        output_path=db_path.parent,
        global_position_indices=[0],
    )
    return db_path


@pytest.fixture
def label_tiff(tmp_path: Path) -> Path:
    """A simple 2D label TIFF."""
    arr = np.zeros((256, 256), dtype=np.uint16)
    arr[10:30, 10:30] = 1
    p = tmp_path / "A1_0000_labels.tif"
    tifffile.imwrite(p, arr)
    return p


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
    # Get the first FOV name from _well_fovs
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
