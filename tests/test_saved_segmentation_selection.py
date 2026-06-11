"""Tests for CaliGui._on_saved_segmentation_selected()."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from cali.gui import CaliGui
from cali.gui._detection_gui import CellposeSettingsData
from cali.sqlmodel import DetectionSettings
from cali.sqlmodel._model import SQLModel

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot


@pytest.fixture
def cali_gui(qtbot: QtBot) -> CaliGui:
    """Create a CaliGui instance for testing."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    return gui


def _make_db_with_cellpose_detection(
    tmp_path: Path, model_type: str
) -> tuple[Path, int]:
    """Create a database with a single cellpose DetectionSettings row."""
    from sqlmodel import Session, create_engine

    db_path = tmp_path / "test.cali"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        d = DetectionSettings(
            method="cellpose",
            model_type=model_type,
            custom_model="/some/model/path",
            diameter=30.0,
            cellprob_threshold=-1.0,
            flow_threshold=0.6,
            min_size=20,
            normalize=False,
            batch_size=4,
            use_gpu=False,
        )
        session.add(d)
        session.commit()
        detection_id = d.id
    engine.dispose(close=True)
    assert detection_id is not None
    return db_path, detection_id


def test_on_saved_segmentation_selected_no_database(
    cali_gui: CaliGui, qtbot: QtBot
) -> None:
    """With no database loaded, the handler should do nothing."""
    assert cali_gui._database_path is None

    # Should not raise even though there is no database
    cali_gui._on_saved_segmentation_selected(1)


def test_on_saved_segmentation_selected_loads_cellpose_settings(
    cali_gui: CaliGui, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting a saved cellpose segmentation loads it into the detection widget."""
    cellpose_wdg = cali_gui._detection_wdg._cellpose_wdg
    model_type = cellpose_wdg._models_combo.itemText(0)

    db_path, detection_id = _make_db_with_cellpose_detection(tmp_path, model_type)
    cali_gui._database_path = str(db_path)

    refresh = MagicMock()
    monkeypatch.setattr(cali_gui, "_on_fov_table_selection_changed", refresh)

    cali_gui._on_saved_segmentation_selected(detection_id)

    refresh.assert_called_once()
    value = cali_gui._detection_wdg.value()
    assert isinstance(value, CellposeSettingsData)
    assert value.model_type == model_type
    assert value.diameter == 30.0
    assert value.cellprob_threshold == -1.0
    assert value.flow_threshold == 0.6
    assert value.min_size == 20
    assert value.normalize is False
    assert value.batch_size == 4
    assert value.use_gpu is False


def test_on_saved_segmentation_selected_invalid_id_shows_error(
    cali_gui: CaliGui, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown detection_settings_id should surface an error dialog."""
    db_path, _ = _make_db_with_cellpose_detection(tmp_path, "cpsam")
    cali_gui._database_path = str(db_path)

    error_dialog = MagicMock()
    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", error_dialog)

    cali_gui._on_saved_segmentation_selected(999)

    error_dialog.assert_called_once()
