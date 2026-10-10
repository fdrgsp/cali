"""Explicit, background, verified CASCADE downloads and safe GUI lifecycle."""

from __future__ import annotations

from threading import Event, get_ident
from typing import TYPE_CHECKING, Any

import pytest
from qtpy.QtCore import Qt, QTimer

from cali import _cascade_models as models
from cali.gui import CaliGui
from cali.gui._extraction_gui import _ExtractionGUI

from .test_cascade_models import NAME, _archive, _config
from .test_cascade_models import fake_download as fake_download

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot


@pytest.fixture
def extraction(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _ExtractionGUI:
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(tmp_path / "model cache"))
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    widget.resize(750, 900)
    widget.show()
    widget._settings_tabs.setCurrentIndex(1)
    widget._spike_outputs.setValue(("cascade",), NAME, "mps")
    return widget


def test_explicit_download_is_background_verified_and_reuses_cache(
    extraction: _ExtractionGUI,
    qtbot: QtBot,
    fake_download: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = extraction._spike_outputs
    threads: list[int] = []
    original_fetch = models._fetch_url

    def fetch(*args: Any, **kwargs: Any) -> None:
        threads.append(get_ident())
        original_fetch(*args, **kwargs)

    monkeypatch.setattr(models, "_fetch_url", fetch)
    assert fake_download["calls"] == []
    assert not models.cascade_model_dir().exists()
    assert "Weights not downloaded" in outputs._cache_info.text()
    qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert len(fake_download["calls"]) == 2

    assert threads and all(thread != get_ident() for thread in threads)
    assert models.load_cascade_model(NAME).minimum_frames == 65
    assert "Verified" in outputs._download_status.text()
    assert "200 ms" in outputs._info.text()
    assert "Cache last verified" in outputs._cache_info.text()
    assert str(models.cascade_model_dir() / NAME) in outputs._cache_info.text()
    assert outputs._model.currentText() == NAME
    assert outputs.methods() == ("cascade",)
    assert outputs._device.currentData() == "mps"
    assert outputs._download.isEnabled()
    assert outputs._download_progress.isHidden()
    qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert len(fake_download["calls"]) == 2

    def damaged_cache(*args: Any, **kwargs: Any) -> models.CascadeModel:
        raise models.CascadeModelError("checkpoint checksum changed")

    monkeypatch.setattr(models, "load_cascade_model", damaged_cache)
    qtbot.mouseClick(outputs._verify, Qt.MouseButton.LeftButton)
    assert "checkpoint checksum changed" in outputs._info.text()
    assert "Local model files found" in outputs._cache_info.text()
    assert "last verified" not in outputs._cache_info.text()


@pytest.fixture
def paused_download(
    monkeypatch: pytest.MonkeyPatch, fake_download: dict[str, Any]
) -> tuple[Event, Event]:
    entered, release = Event(), Event()
    original_fetch = models._fetch_url

    def fetch(url: str, target: Path, limit: int, **kwargs: Any) -> None:
        if url != models.CATALOGUE_URL:
            entered.set()
            assert release.wait(timeout=10), "Test did not release its paused download"
        original_fetch(url, target, limit, **kwargs)

    monkeypatch.setattr(models, "_fetch_url", fetch)
    return entered, release


def test_download_keeps_gui_responsive_and_preserves_newer_settings(
    extraction: _ExtractionGUI,
    qtbot: QtBot,
    fake_download: dict[str, Any],
    paused_download: tuple[Event, Event],
) -> None:
    outputs = extraction._spike_outputs
    entered, release = paused_download
    try:
        qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
        heartbeat = Event()
        QTimer.singleShot(0, heartbeat.set)
        qtbot.waitUntil(lambda: entered.is_set() and heartbeat.is_set(), timeout=5000)
        assert outputs.is_downloading()
        assert outputs._download_progress.isVisible()
        assert not outputs._download.isEnabled()
        assert not outputs._model.isEnabled()
        with pytest.raises(ValueError, match="download to finish"):
            extraction.to_model_settings()
        outputs._download_model()  # A second click cannot create another job.
        outputs.setValue(("oasis", "cascade"), "another-model", "cpu")
        outputs.setReadOnly(True)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert outputs._model.currentText() == "another-model"
    assert outputs.methods() == ("oasis", "cascade")
    assert outputs._device.currentData() == "cpu"
    assert NAME in outputs._download_status.text()
    assert "Verified" not in outputs._info.text()
    assert "Weights not downloaded" in outputs._cache_info.text()
    assert "another-model" in outputs._cache_info.text()
    assert not outputs._download.isEnabled()
    assert not outputs._cascade.isEnabled()
    assert len(fake_download["calls"]) == 2


def test_failed_download_preserves_selection_and_can_retry(
    extraction: _ExtractionGUI, qtbot: QtBot, fake_download: dict[str, Any]
) -> None:
    outputs = extraction._spike_outputs
    fake_download["archive"] = _archive(_config(sampling_rate=0))
    qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert "Download failed" in outputs._download_status.text()
    assert "Weights not downloaded" in outputs._cache_info.text()
    assert not (models.cascade_model_dir() / NAME).exists()
    assert not list(models.cascade_model_dir().glob(".*"))
    assert outputs.methods() == ("cascade",)
    assert outputs._model.currentText() == NAME
    assert outputs._download.isEnabled()
    fake_download["archive"] = _archive()
    qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert "Verified" in outputs._download_status.text()
    assert len(fake_download["calls"]) == 3


def test_cancelled_download_leaves_no_partial_model_and_can_retry(
    extraction: _ExtractionGUI,
    qtbot: QtBot,
    fake_download: dict[str, Any],
    paused_download: tuple[Event, Event],
) -> None:
    outputs = extraction._spike_outputs
    entered, release = paused_download
    try:
        qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        qtbot.mouseClick(outputs._cancel_download_btn, Qt.MouseButton.LeftButton)
        assert "Cancelling" in outputs._download_status.text()
        assert not outputs._cancel_download_btn.isEnabled()
        assert not outputs._download.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert "Cancelled" in outputs._download_status.text()
    assert "Weights not downloaded" in outputs._cache_info.text()
    assert not (models.cascade_model_dir() / NAME).exists()
    assert not list(models.cascade_model_dir().glob(".*"))
    assert outputs.methods() == ("cascade",)
    assert outputs._download.isEnabled()
    qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert "Verified" in outputs._download_status.text()


def test_destroying_widget_cancels_download_without_gui_callbacks(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_download: dict[str, Any],
    paused_download: tuple[Event, Event],
) -> None:
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(tmp_path / "cache"))
    # Explicit destruction owns cleanup; qtbot must not close a deleted wrapper.
    extraction = _ExtractionGUI(cascade_enabled=True)
    extraction._settings_tabs.setCurrentIndex(1)
    extraction._spike_outputs.setValue(("cascade",), NAME, "cpu")
    extraction.show()
    outputs = extraction._spike_outputs
    entered, release = paused_download
    try:
        qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        worker = outputs._download_worker
        assert worker is not None
        with qtbot.waitSignal(extraction.destroyed, timeout=1000):
            extraction.deleteLater()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not worker.is_running, timeout=5000)
    assert not (models.cascade_model_dir() / NAME).exists()
    assert not list(models.cascade_model_dir().glob(".*"))


def test_closing_main_window_cancels_its_download(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_download: dict[str, Any],
    paused_download: tuple[Event, Event],
) -> None:
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(tmp_path / "cache"))
    gui = CaliGui(cascade_gui_enabled=True)
    qtbot.addWidget(gui)
    outputs = gui._extraction_wdg._spike_outputs
    outputs.setValue(("cascade",), NAME, "cpu")
    entered, release = paused_download
    try:
        outputs._download_model()
        qtbot.waitUntil(entered.is_set, timeout=5000)
        gui.close()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=5000)
    assert "Cancelled" in outputs._download_status.text()
    assert not (models.cascade_model_dir() / NAME).exists()
    assert not list(models.cascade_model_dir().glob(".*"))


@pytest.mark.parametrize("state", ["gate", "readonly", "oasis", "empty"])
def test_inactive_download_controls_cannot_fetch(
    state: str,
    qtbot: QtBot,
    tmp_path: Path,
    fake_download: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(root))
    widget = _ExtractionGUI(cascade_enabled=state != "gate")
    qtbot.addWidget(widget)
    outputs = widget._spike_outputs
    outputs.setValue(("oasis",) if state == "oasis" else ("cascade",), NAME, "cpu")
    if state == "readonly":
        outputs.setReadOnly(True)
    if state == "empty":
        outputs._model.setCurrentText("")
    assert not outputs._download.isEnabled()
    outputs._download_model()
    assert not outputs.is_downloading()
    assert fake_download["calls"] == []
    assert not root.exists()
