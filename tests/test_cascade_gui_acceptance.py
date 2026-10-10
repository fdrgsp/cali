"""Run-button acceptance and worker lifecycle, including optional pretrained runs."""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QStyle, QStyleOptionGroupBox
from sqlmodel import Session, select

from cali import _cascade_models as models
from cali.gui import CaliGui
from cali.gui._analysis_gui import AnalysisSettingsData, SpikeData
from cali.gui._extraction_gui import (
    ExtractionSettingsData,
    MetadataData,
    TraceExtractionData,
)
from cali.sqlmodel import (
    CaliResult,
    SpikeAnalysisSettings,
    SpikeInferenceRun,
    Traces,
    create_cali_engine,
)

from .test_cascade_extraction import _dataset
from .test_cascade_reference import fake_reference as fake_reference
from .test_cascade_runner_analysis import _seed

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pytestqt.qtbot import QtBot


@pytest.fixture(autouse=True)
def offline_icons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep installed-wheel GUI tests offline even with --noconftest."""
    icon = tmp_path / "icon.svg"
    icon.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<rect width="24" height="24"/></svg>'
    )
    monkeypatch.setattr("pyconify.api.svg_path", lambda *a, **kw: icon)
    monkeypatch.setattr("superqt.iconify.svg_path", lambda *a, **kw: icon)


@pytest.mark.parametrize("methods", [("oasis",), ("cascade",), ("oasis", "cascade")])
def test_run_button_extracts_and_reanalyzes_offline(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_reference: tuple,
    methods: tuple,
) -> None:
    _exercise_run_button(qtbot, tmp_path, monkeypatch, methods)


@pytest.mark.parametrize("methods", [("cascade",), ("oasis", "cascade")])
def test_run_button_routes_missing_model_to_selector(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    methods: tuple,
) -> None:
    path = tmp_path / "gui.cali"
    _, detection, _, _ = _seed(path, methods)
    errors = Mock()
    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", errors)
    gui = CaliGui(cascade_gui_enabled=True)
    qtbot.addWidget(gui)
    gui._database_path, gui._output_path = str(path), str(tmp_path)
    gui._data = _dataset(count=256, rate=10)
    gui._runs_panel.set_database_path(path)
    start_worker = Mock()
    monkeypatch.setattr(gui, "_start_run_worker", start_worker)
    run = gui._run_cali_wdg
    run._detection_settings_combo.addItem("Seeded masks", detection)
    run._detection_settings_combo.setCurrentIndex(
        run._detection_settings_combo.count() - 1
    )
    run._run_options_combo.setCurrentText("Extraction and Analysis (require detection)")
    # An unfinished GUI selection is allowed; saved/headless settings stay strict.
    gui._extraction_wdg._spike_outputs.setValue(methods, None, "cpu")
    gui._extraction_wdg._settings_tabs.setCurrentIndex(2)
    gui._sub_tab.setCurrentWidget(gui._analysis_tab)
    gui.show()
    gui.activateWindow()
    qtbot.waitUntil(gui.isActiveWindow)
    qtbot.mouseClick(run._run_btn, Qt.MouseButton.LeftButton)

    errors.assert_called_once()
    message = errors.call_args.args[1]
    assert "Extract traces → Spike inference" in message
    assert "Prepare traces" in message
    assert "Download model" in message
    assert "validation error" not in message
    start_worker.assert_not_called()
    assert gui._run_worker is None
    assert gui._extraction_wdg._spike_outputs.methods() == methods
    assert gui._sub_tab.currentWidget() is gui._extraction_tab
    assert gui._extraction_wdg._settings_tabs.currentIndex() == 1
    qtbot.waitUntil(gui._extraction_wdg._spike_outputs._model.hasFocus)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        assert not session.exec(select(Traces)).all()
    engine.dispose()


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Pretrained GUI acceptance runs in the optional installed-wheel job",
)
def test_pretrained_run_button_extracts_and_reanalyzes_offline(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_run_button(
        qtbot, tmp_path, monkeypatch, ("oasis", "cascade"), pretrained=True
    )


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Pretrained first-use workflow runs in the optional installed-wheel job",
)
@pytest.mark.parametrize("methods", [("cascade",), ("oasis", "cascade")])
def test_fresh_cache_download_then_pretrained_run_and_offline_reanalysis(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    methods: tuple,
) -> None:
    """Use real pinned weights through GUI download, extraction and saved reuse.

    Only the HTTPS transfer is replaced with local bytes; catalogue validation,
    archive publication, model checks and Torch inference run normally.
    """
    metadata = json.loads(
        (Path(__file__).parent / "fixtures/cascade_reference/manifest.json").read_text()
    )
    source = models.load_cascade_model(
        metadata["model_name"],
        expected_manifest=metadata["model_manifest_sha256"],
    )
    entry = next(
        item
        for item in models.get_cascade_catalogue(allow_bundled=True)
        if item.name == source.name
    )
    archive = tmp_path / "pretrained.zip"
    with ZipFile(archive, "w") as bundle:
        for filename in ("config.yaml", *source.weight_files):
            bundle.write(source.directory / filename, f"{source.name}/{filename}")
    cache = tmp_path / "fresh model cache"
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(cache))
    transfers: list[tuple[str, int]] = []

    def fetch(url: str, target: Path, limit: int, **kwargs: Any) -> None:
        transfers.append((url, threading.get_ident()))
        models._check_download_cancelled(kwargs.get("cancel_requested"))
        if url == models.CATALOGUE_URL:
            target.write_bytes(
                (
                    Path(models.__file__).parent / "resources/cascade_catalogue.yaml"
                ).read_bytes()
            )
        else:
            assert url == entry.url
            shutil.copyfile(archive, target)
        assert target.stat().st_size <= limit

    monkeypatch.setattr(models, "_fetch_url", fetch)

    def choose_and_download(gui: CaliGui, name: str) -> None:
        extraction = gui._extraction_wdg
        outputs = extraction._spike_outputs
        assert not cache.exists()
        assert outputs._model.currentText() == ""
        assert not outputs._download.isEnabled()
        extraction._metadata_wdg._frame_rate_spin.setValue(30)
        assert transfers == []
        assert not cache.exists()
        index = outputs._model.findText(name)
        assert index > 0  # A real offered choice, following the explicit blank item.
        gui.show()
        gui._sub_tab.setCurrentWidget(gui._extraction_tab)
        extraction._settings_tabs.setCurrentIndex(1)
        if "oasis" in methods:
            option = QStyleOptionGroupBox()
            outputs._oasis.initStyleOption(option)
            checkbox = outputs._oasis.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox,
                option,
                QStyle.SubControl.SC_GroupBoxCheckBox,
                outputs._oasis,
            )
            qtbot.mouseClick(
                outputs._oasis, Qt.MouseButton.LeftButton, pos=checkbox.center()
            )
        assert outputs.methods() == methods
        outputs._model.setCurrentIndex(index)
        assert outputs._download.isEnabled()
        assert "Weights not downloaded" in outputs._cache_info.text()
        qtbot.mouseClick(outputs._download, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: not outputs.is_downloading(), timeout=10000)
        assert [url for url, _ in transfers] == [
            models.CATALOGUE_URL,
            entry.url,
        ]
        assert all(ident != threading.get_ident() for _, ident in transfers)
        downloaded = models.load_cascade_model(
            name, expected_manifest=source.manifest_sha256
        )
        assert downloaded.directory == cache / name
        assert "Cache last verified" in outputs._cache_info.text()
        assert not list(cache.glob(".*"))
        # Both extraction and later database-only analysis must now work offline.
        monkeypatch.setattr(
            models, "_fetch_url", Mock(side_effect=AssertionError("No more downloads"))
        )

    _exercise_run_button(
        qtbot,
        tmp_path,
        monkeypatch,
        methods,
        pretrained=True,
        prepare_model=choose_and_download,
    )


def _exercise_run_button(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    methods: tuple,
    *,
    pretrained: bool = False,
    prepare_model: Callable[[CaliGui, str], None] | None = None,
) -> None:
    path = tmp_path / "gui.cali"
    _, detection, _, _ = _seed(path, methods, pretrained=pretrained)
    rate = 30 if pretrained else 10
    dataset = _dataset(count=256, rate=rate)
    dataset.position_metadata.return_value = (256, dataset.isel.return_value[1])
    dataset.sequence = Mock()
    dataset.sequence.stage_positions = [Mock()]
    monkeypatch.setattr(
        "cali.runner._cali_runner.load_data_from_path", lambda *a: dataset
    )
    errors = Mock()
    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", errors)
    gui = CaliGui(cascade_gui_enabled=True)
    qtbot.addWidget(gui)
    gui._database_path, gui._output_path, gui._data = str(path), str(tmp_path), dataset
    gui._runs_panel.set_database_path(path)
    monkeypatch.setattr(gui, "_save_plate_map_to_database", lambda: None)
    run = gui._run_cali_wdg
    run._detection_settings_combo.addItem("Seeded masks", detection)
    run._detection_settings_combo.setCurrentIndex(
        run._detection_settings_combo.count() - 1
    )
    run._run_options_combo.setCurrentText("Extraction and Analysis (require detection)")
    model = "Test_10Hz"
    if pretrained:
        model = json.loads(
            (
                Path(__file__).parent / "fixtures/cascade_reference/manifest.json"
            ).read_text()
        )["model_name"]
    if prepare_model is not None:
        prepare_model(gui, model)
    gui._extraction_wdg.setValue(
        ExtractionSettingsData(
            metadata_data=MetadataData(frame_rate=rate),
            trace_extraction_data=TraceExtractionData(
                spike_methods=methods,
                cascade_model=model if "cascade" in methods else None,
                cascade_device="cpu",
                discard_initial_value=11,
                dff_window_size=5,
            ),
            threads=1,
        )
    )
    children = tuple(
        SpikeData.from_model(
            SpikeAnalysisSettings(
                method=method,
                threshold_mode="global",
                threshold_value=0.001 if method == "oasis" else 0.2,
                ccg_n_shuffles=2,
            )
        )
        for method in methods
    )
    gui._analysis_wdg.setValue(
        AnalysisSettingsData(spike_settings=children, n_processes=1, frame_rate=rate)
    )
    gui._extraction_wdg._export_group.setChecked(False)
    gui._analysis_wdg._export_group.setChecked(False)
    gui._sub_tab.setCurrentIndex(1)
    # GUI getters must run on the main thread; runner loading must not.
    main_thread = threading.get_ident()
    for widget in (gui._extraction_wdg, gui._analysis_wdg):
        original = widget.get_export_options

        def exports(original: Callable = original) -> Any:
            assert threading.get_ident() == main_thread
            return original()

        monkeypatch.setattr(widget, "get_export_options", exports)
    calls = []
    original_run = gui._runner.run

    def run_in_thread(*args: Any, **kwargs: Any) -> Any:
        calls.append(threading.get_ident())
        return original_run(*args, **kwargs)

    monkeypatch.setattr(gui._runner, "run", run_in_thread)
    qtbot.mouseClick(run._run_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui._run_worker is None, timeout=60000)
    errors.assert_not_called()
    assert calls and all(ident != main_thread for ident in calls)
    assert "Finished" in run._progress_pos_label.text()
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        owner = session.exec(
            select(CaliResult).where(CaliResult.extraction_settings_id.is_not(None))
        ).all()[-1]
        source_id, extraction_id = owner.id, owner.extraction_settings_id
        traces = session.exec(
            select(Traces).where(Traces.analysis_result_id == source_id)
        ).all()
        before = [list(trace.raw_trace) for trace in traces]
        assert len(traces) == 2
        assert all(len(values) == 245 for values in before)
        inferences = session.exec(
            select(SpikeInferenceRun).where(
                SpikeInferenceRun.extraction_result_id == source_id
            )
        ).all()
        assert {row.method for row in inferences} == set(methods)
        spike_outputs = [
            [
                (child.inference_run.method, list(child.values))
                for child in trace.spike_traces
            ]
            for trace in traces
        ]
        if pretrained:
            cascade = next(row for row in inferences if row.method == "cascade")
            assert (
                cascade.weights_manifest_sha256
                == json.loads(
                    (
                        Path(__file__).parent
                        / "fixtures/cascade_reference/manifest.json"
                    ).read_text()
                )["model_manifest_sha256"]
            )
    # Database-only analysis must use saved outputs and never load the model/package.
    gui._data = None
    gui._populate_settings(str(path))
    run._run_options_combo.setCurrentText(
        "Analysis Only (require detection and extraction)"
    )
    run._detection_settings_combo.setCurrentIndex(
        run._detection_settings_combo.findData(detection)
    )
    run._extraction_settings_combo.setCurrentIndex(
        run._extraction_settings_combo.findData(extraction_id)
    )
    monkeypatch.setattr(
        "cali.extraction._spike_inference._cascade_reference.load_cascade_model",
        Mock(side_effect=AssertionError("No model during reanalysis")),
    )
    monkeypatch.setattr(
        "cali.extraction._spike_inference._cascade_reference.load_cascade_package",
        Mock(side_effect=AssertionError("No package during reanalysis")),
    )
    changed = tuple(
        SpikeData.from_model(
            SpikeAnalysisSettings(
                method=method,
                threshold_mode="global",
                threshold_value=0.002 if method == "oasis" else 0.22,
                ccg_n_shuffles=2,
            )
        )
        for method in methods
    )
    gui._analysis_wdg.setValue(
        AnalysisSettingsData(spike_settings=changed, n_processes=1, frame_rate=rate)
    )
    qtbot.mouseClick(run._run_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui._run_worker is None, timeout=60000)
    errors.assert_not_called()
    with Session(engine) as session:
        traces = session.exec(
            select(Traces).where(Traces.analysis_result_id == source_id)
        ).all()
        assert [list(trace.raw_trace) for trace in traces] == before
        assert [
            [
                (child.inference_run.method, list(child.values))
                for child in trace.spike_traces
            ]
            for trace in traces
        ] == spike_outputs
        assert len(session.exec(select(SpikeInferenceRun)).all()) == len(methods)
    engine.dispose()


@pytest.mark.parametrize("outcome", ["error", "cancel", "close"])
def test_worker_outcome_and_close_lifecycle(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    gui = CaliGui(cascade_gui_enabled=True)
    qtbot.addWidget(gui)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def paused(**kwargs: Any) -> Generator[str, None, None]:
        entered.set()
        assert release.wait(5)
        try:
            yield "Ready"
            if outcome == "error":
                raise ValueError("Selected CASCADE model is unavailable")
        finally:
            closed.set()

    monkeypatch.setattr(gui._runner, "run", paused)
    cancel = Mock()
    monkeypatch.setattr(gui._runner, "cancel", cancel)
    errors = Mock()
    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", errors)
    gui._start_run_worker({})
    heartbeat = threading.Event()
    QTimer.singleShot(0, heartbeat.set)
    try:
        qtbot.waitUntil(lambda: entered.is_set() and heartbeat.is_set(), timeout=5000)
        gui._start_run_worker({})  # Duplicate launch is ignored.
        if outcome == "close":
            gui.close()
            assert gui.isVisible()
        elif outcome == "cancel":
            gui._on_cali_cancel()
        assert not gui._run_cali_wdg._run_btn.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: gui._run_worker is None, timeout=5000)
    assert closed.is_set()
    if outcome == "error":
        assert "Failed" in gui._run_cali_wdg._progress_pos_label.text()
        assert "unavailable" in errors.call_args.args[1]
    else:
        errors.assert_not_called()
        cancel.assert_called()
        if outcome == "close":
            assert not gui.isVisible()
        else:
            assert "Cancelled" in gui._run_cali_wdg._progress_pos_label.text()
