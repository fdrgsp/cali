"""Tests for _DetectionGUI and _ImportedLabelsWidget new functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cali.gui._detection_gui import _DetectionGUI, _ImportedLabelsWidget

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def detection_gui(qtbot: QtBot) -> _DetectionGUI:
    wdg = _DetectionGUI()
    qtbot.addWidget(wdg)
    return wdg


@pytest.fixture
def imported_wdg(qtbot: QtBot) -> _ImportedLabelsWidget:
    wdg = _ImportedLabelsWidget()
    qtbot.addWidget(wdg)
    return wdg


# ---------------------------------------------------------------------------
# _DetectionGUI.active_method
# ---------------------------------------------------------------------------


def test_active_method_defaults_to_cellpose(detection_gui: _DetectionGUI) -> None:
    assert detection_gui.active_method() == "cellpose"


def test_active_method_after_set_imported(detection_gui: _DetectionGUI) -> None:
    detection_gui.setValue(method="imported_labels")
    assert detection_gui.active_method() == "imported_labels"


def test_active_method_after_set_cellpose(detection_gui: _DetectionGUI) -> None:
    detection_gui.setValue(method="imported_labels")
    detection_gui.setValue(method="cellpose")
    assert detection_gui.active_method() == "cellpose"


# ---------------------------------------------------------------------------
# _DetectionGUI.setValue with method parameter
# ---------------------------------------------------------------------------


def test_set_value_imported_unchecks_cellpose(detection_gui: _DetectionGUI) -> None:
    detection_gui.setValue(method="imported_labels")
    assert detection_gui._imported_labels_wdg.isChecked()
    assert not detection_gui._cellpose_wdg.isChecked()


def test_set_value_cellpose_unchecks_imported(detection_gui: _DetectionGUI) -> None:
    detection_gui.setValue(method="imported_labels")
    detection_gui.setValue(method="cellpose")
    assert detection_gui._cellpose_wdg.isChecked()
    assert not detection_gui._imported_labels_wdg.isChecked()


# ---------------------------------------------------------------------------
# Mutual exclusion toggle logic
# ---------------------------------------------------------------------------


def test_checking_imported_unchecks_cellpose(detection_gui: _DetectionGUI) -> None:
    detection_gui._imported_labels_wdg.setChecked(True)
    assert not detection_gui._cellpose_wdg.isChecked()


def test_checking_cellpose_unchecks_imported(detection_gui: _DetectionGUI) -> None:
    detection_gui._imported_labels_wdg.setChecked(True)
    detection_gui._cellpose_wdg.setChecked(True)
    assert not detection_gui._imported_labels_wdg.isChecked()


def test_unchecking_cellpose_without_imported_rechecks_cellpose(
    detection_gui: _DetectionGUI,
) -> None:
    """Cannot have both unchecked - cellpose re-checks itself."""
    # Both start: cellpose=True, imported=False
    # Unchecking cellpose via toggle should re-check it
    detection_gui._on_cellpose_toggled(False)
    assert detection_gui._cellpose_wdg.isChecked()


def test_unchecking_imported_without_cellpose_rechecks_imported(
    detection_gui: _DetectionGUI,
) -> None:
    """Cannot have both unchecked - imported re-checks itself."""
    detection_gui.setValue(method="imported_labels")
    detection_gui._on_imported_toggled(False)
    assert detection_gui._imported_labels_wdg.isChecked()


# ---------------------------------------------------------------------------
# _DetectionGUI.enable
# ---------------------------------------------------------------------------


def test_enable_disables_both_widgets(detection_gui: _DetectionGUI) -> None:
    detection_gui.enable(False)
    assert not detection_gui._cellpose_wdg.isEnabled()
    assert not detection_gui._imported_labels_wdg.isEnabled()

    detection_gui.enable(True)
    assert detection_gui._cellpose_wdg.isEnabled()
    assert detection_gui._imported_labels_wdg.isEnabled()


# ---------------------------------------------------------------------------
# _DetectionGUI.reset
# ---------------------------------------------------------------------------


def test_reset_restores_cellpose_active(detection_gui: _DetectionGUI) -> None:
    detection_gui.setValue(method="imported_labels")
    detection_gui.reset()
    assert detection_gui.active_method() == "cellpose"
    assert detection_gui._cellpose_wdg.isChecked()
    assert not detection_gui._imported_labels_wdg.isChecked()


# ---------------------------------------------------------------------------
# _DetectionGUI.to_model_settings
# ---------------------------------------------------------------------------


def test_to_model_settings_cellpose(detection_gui: _DetectionGUI) -> None:
    ds = detection_gui.to_model_settings()
    assert ds.method == "cellpose"


def test_to_model_settings_imported_labels(detection_gui: _DetectionGUI) -> None:
    detection_gui.setValue(method="imported_labels")
    ds = detection_gui.to_model_settings()
    assert ds.method == "imported_labels"


# ---------------------------------------------------------------------------
# _ImportedLabelsWidget
# ---------------------------------------------------------------------------


def test_imported_widget_defaults(imported_wdg: _ImportedLabelsWidget) -> None:
    assert imported_wdg.isCheckable()
    assert not imported_wdg.isChecked()
    assert imported_wdg.detection_settings_id() is None
    assert imported_wdg._status_label.text() == "No labels imported yet."


def test_set_database_path(imported_wdg: _ImportedLabelsWidget) -> None:
    imported_wdg.set_database_path("/tmp/test.cali")
    assert imported_wdg._database_path == "/tmp/test.cali"


def test_set_database_path_none(imported_wdg: _ImportedLabelsWidget) -> None:
    imported_wdg.set_database_path("/tmp/test.cali")
    imported_wdg.set_database_path(None)
    assert imported_wdg._database_path is None


@pytest.mark.parametrize("det_id", [1, 42, 100])
def test_set_detection_settings_id(
    imported_wdg: _ImportedLabelsWidget, det_id: int
) -> None:
    imported_wdg.set_detection_settings_id(det_id)
    assert imported_wdg.detection_settings_id() == det_id
    assert str(det_id) in imported_wdg._status_label.text()


def test_set_detection_settings_id_none_resets_label(
    imported_wdg: _ImportedLabelsWidget,
) -> None:
    imported_wdg.set_detection_settings_id(5)
    imported_wdg.set_detection_settings_id(None)
    assert imported_wdg.detection_settings_id() is None
    assert imported_wdg._status_label.text() == "No labels imported yet."


def test_reset_imported_widget(imported_wdg: _ImportedLabelsWidget) -> None:
    imported_wdg.set_detection_settings_id(10)
    imported_wdg._n_imported_fovs = 5
    imported_wdg.reset()
    assert imported_wdg.detection_settings_id() is None
    assert imported_wdg._n_imported_fovs == 0
    assert imported_wdg._status_label.text() == "No labels imported yet."


def test_import_clicked_no_database(
    imported_wdg: _ImportedLabelsWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clicking import without a database path shows an error dialog."""
    errors: list[str] = []
    monkeypatch.setattr(
        "cali.gui._detection_gui.show_error_dialog",
        lambda parent, msg: errors.append(msg),
    )
    imported_wdg._database_path = None
    imported_wdg._on_import_clicked()
    assert len(errors) == 1
    assert "database" in errors[0].lower()
