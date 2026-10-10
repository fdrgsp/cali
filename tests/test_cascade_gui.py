"""CASCADE GUI ownership, compatibility, gate and stored-results integration."""

from __future__ import annotations

import json
import math
import shlex
import sys
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMessageBox, QWidget
from sqlmodel import Session

from cali.gui import CaliGui
from cali.gui._analysis_gui import AnalysisSettingsData, SpikeData, _AnalysisGUI
from cali.gui._extraction_gui import (
    ExtractionSettingsData,
    MetadataData,
    TraceExtractionData,
    _ExtractionGUI,
)
from cali.gui._pygraph_plot_widgets import (
    _MultilWellGraphWidget,
    _SingleWellGraphWidget,
)
from cali.sqlmodel import (
    AnalysisSettings,
    CaliResult,
    ExtractionSettings,
    SpikeAnalysisSettings,
)

from .test_spike_export import spike_database as spike_database

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot


@pytest.mark.parametrize("enabled", [False, True])
def test_new_and_reset_gui_outputs_respect_gate(qtbot: QtBot, enabled: bool) -> None:
    widget = _ExtractionGUI(cascade_enabled=enabled)
    qtbot.addWidget(widget)
    outputs = widget._spike_outputs
    expected = ("cascade",) if enabled else ("oasis",)
    assert outputs.methods() == expected
    assert outputs._cascade.isEnabled() == enabled
    assert widget._trace_extraction_wdg._decay_constant_spin.isEnabled()
    assert widget._trace_extraction_wdg._frame_rate_verified.isEnabled() == enabled
    outputs.setValue(("oasis", "cascade"), "missing-model", "cpu")
    assert outputs.methods() == ("oasis", "cascade")
    widget.reset()
    assert outputs.methods() == expected
    assert widget.value().trace_extraction_data.discard_initial_value == 0
    assert widget.value().trace_extraction_data.discard_initial_unit == "frames"
    assert ExtractionSettings().spike_methods == ("oasis",)


def test_loaded_cascade_stays_checked_and_blocks_gated_extraction(qtbot: QtBot) -> None:
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)
    data = TraceExtractionData(
        spike_methods=("cascade",),
        cascade_model="unavailable-model",
        cascade_device="mps",
    )
    widget.setValue(ExtractionSettingsData(trace_extraction_data=data))
    assert widget.value().trace_extraction_data == data
    with pytest.raises(ValueError, match="release checks"):
        widget.to_model_settings()
    assert widget._spike_outputs._cascade.isChecked()
    assert not widget._spike_outputs._oasis.isChecked()


@pytest.mark.parametrize("methods", [("oasis",), ("cascade",), ("oasis", "cascade")])
def test_extraction_model_and_json_roundtrip(qtbot: QtBot, methods: tuple) -> None:
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    data = TraceExtractionData(
        spike_methods=methods,
        cascade_model="chosen-model" if "cascade" in methods else None,
        cascade_device="cpu" if "cascade" in methods else "auto",
        discard_initial_value=1.25,
        discard_initial_unit="seconds",
        frame_rate_verified=True,
        frame_rate=30,
    )
    loaded = TraceExtractionData.from_json(json.loads(json.dumps(asdict(data))))
    widget.setValue(
        ExtractionSettingsData(
            trace_extraction_data=loaded, metadata_data=MetadataData(frame_rate=30)
        )
    )
    assert widget.value().trace_extraction_data == data
    model = widget.to_model_settings()
    assert model.spike_methods == methods
    assert model.cascade_model == data.cascade_model
    assert model.cascade_device == data.cascade_device
    assert model.discard_initial_value == 1.25
    assert model.frame_rate_verified


@pytest.mark.parametrize(
    "legacy", [{}, {"spike_method": "oasis"}, {"spike_methods": ["oasis"]}]
)
def test_old_extraction_json_stays_oasis(legacy: dict) -> None:
    settings = TraceExtractionData.from_json(legacy)
    assert settings.spike_methods == ("oasis",)
    assert settings.discard_initial_value == 0
    assert settings.discard_initial_unit == "frames"
    assert "spike_method" not in asdict(settings)


@pytest.mark.parametrize(
    "bad",
    [
        {"spike_methods": []},
        {"spike_methods": ["unknown"]},
        {"spike_method": None},
        {"spike_method": "oasis", "spike_methods": ["cascade"]},
        {"discard_initial_unit": "minutes"},
        {"discard_initial_value": -1},
        {"discard_initial_value": 0.5},
        {"discard_initial_value": float("nan")},
    ],
)
def test_bad_extraction_json_is_rejected_before_qt_clamps(bad: dict) -> None:
    with pytest.raises(ValueError):
        TraceExtractionData.from_json(bad)


def test_outputs_require_one_and_never_disable_oasis_denoising(qtbot: QtBot) -> None:
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    outputs = widget._spike_outputs
    outputs._cascade.setChecked(False)
    assert outputs.methods() == ("cascade",)
    outputs._oasis.setChecked(True)
    outputs._cascade.setChecked(False)
    assert outputs.methods() == ("oasis",)
    assert not outputs._model.isEnabled()
    assert not outputs._device.isEnabled()
    assert widget._trace_extraction_wdg._decay_constant_spin.isEnabled()


def test_model_catalogue_is_offline_rate_filtered_and_keeps_missing_selection(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cali._cascade_models import CatalogueEntry

    catalogue = Mock(
        return_value=(
            CatalogueEntry(
                "model_10Hz", "https://example.test/ten", "10 Hz model metadata", 10
            ),
            CatalogueEntry(
                "model_30Hz", "https://example.test/thirty", "30 Hz model metadata", 30
            ),
        )
    )
    monkeypatch.setattr("cali._cascade_models.get_cascade_catalogue", catalogue)
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    outputs = widget._spike_outputs
    outputs.setValue(("cascade",), "absent_10Hz", "mps")
    outputs.set_frame_rate(30)
    assert outputs._model.currentText() == "absent_10Hz"
    assert outputs._model.findText("model_10Hz") == -1
    assert outputs._model.findText("model_30Hz") >= 0
    assert all("allow_download" not in call.kwargs for call in catalogue.call_args_list)
    outputs.setValue(("cascade",), None, "cpu")
    outputs.refresh_models()
    assert outputs._model.currentText() == ""
    with pytest.raises(ValueError, match="Choose a CASCADE model"):
        widget.to_model_settings()


@pytest.mark.parametrize("rate", [10, 30, 12])
def test_fresh_install_offers_models_and_explains_download_state(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rate: int,
) -> None:
    root = tmp_path / "no-model-cache"
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(root))
    fetch = Mock(side_effect=AssertionError("Model listing must stay offline"))
    monkeypatch.setattr("cali._cascade_models._fetch_url", fetch)
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    widget._spike_outputs.set_frame_rate(rate)
    outputs = widget._spike_outputs
    assert outputs._model.currentText() == ""
    assert not outputs._download.isEnabled()
    assert "Choose a pretrained model first" in outputs._download.toolTip()
    if rate == 12:
        assert outputs._model.count() == 1
        assert "No pretrained model matches 12 Hz" in outputs._info.text()
        assert "Prepare traces" in outputs._info.text()
    else:
        assert outputs._model.count() > 1
        assert "does not install model weights" in outputs._info.text()
        for index in range(1, outputs._model.count()):
            assert f"_{rate}Hz_" in outputs._model.itemText(index)
        outputs._model.setCurrentIndex(1)
        assert outputs._download.isEnabled()
        assert outputs.value()[1] == outputs._model.currentText()
    assert not root.exists()
    fetch.assert_not_called()


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_install_instructions_target_current_environment_and_pin(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str
) -> None:
    from cali import _cascade_package as package

    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    outputs = widget._spike_outputs
    outputs.setValue(("cascade",), "chosen-model", "mps")
    python = (
        "C:\\Users\\O'Neil\\env space\\python.exe"
        if platform == "win32"
        else "/Users/O'Neil/env space/bin/python"
    )
    cache = tmp_path / "model cache's folder"
    messages: list[str] = []

    def capture_dialog(dialog: QMessageBox) -> int:
        assert dialog.textFormat() == Qt.TextFormat.PlainText
        assert (
            dialog.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
        )
        assert (
            dialog.textInteractionFlags()
            & Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        messages.append(dialog.informativeText())
        return 0

    download = Mock(side_effect=AssertionError("Instructions must remain offline"))
    load = Mock(side_effect=AssertionError("Instructions must not load inference"))
    monkeypatch.setattr("cali._cascade_models.cascade_model_dir", lambda: cache)
    monkeypatch.setattr("cali._cascade_models.download_cascade_model", download)
    monkeypatch.setattr(package, "load_cascade_package", load)
    monkeypatch.setattr(QMessageBox, "exec", capture_dialog)
    monkeypatch.setattr(sys, "executable", python)
    monkeypatch.setattr(sys, "platform", platform)
    outputs._install.click()
    assert len(messages) == 1
    message = messages[0]
    assert "uv sync --extra cascade" in message
    assert "launch with uv run cali" in message
    assert "Restart cali after installing" in message
    install = next(
        line for line in message.splitlines() if package.CASCADE_PACKAGE_URL in line
    )
    download_command = next(
        line for line in message.splitlines() if "cascade-download" in line
    )
    if platform == "win32":
        assert "Windows PowerShell" in message
        assert install.startswith("& 'uv' 'pip' 'install' '--python'")
        assert "'C:\\Users\\O''Neil\\env space\\python.exe'" in install
        assert download_command.startswith(
            "& 'C:\\Users\\O''Neil\\env space\\python.exe' '-m' 'cali'"
        )
        assert "model cache''s folder" in download_command
    else:
        assert shlex.split(install) == [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            f"CascadeTorch @ {package.CASCADE_PACKAGE_URL}",
        ]
        assert shlex.split(download_command) == [
            python,
            "-m",
            "cali",
            "cascade-download",
            "chosen-model",
            "--model-dir",
            str(cache),
        ]
    assert outputs.value() == (("cascade",), "chosen-model", "mps")
    outputs.setValue(("cascade",), None, "mps")
    outputs._install.click()
    assert len(messages) == 2
    assert "<model-name>" in messages[1]
    download.assert_not_called()
    load.assert_not_called()


def test_model_verification_shows_metadata_without_inference_or_fallback(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cali._cascade_models import CascadeModelNotFound

    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    outputs = widget._spike_outputs
    outputs.setValue(("cascade",), "chosen-model", "mps")
    verify = Mock(
        side_effect=CascadeModelNotFound("Run: cali cascade-download chosen-model")
    )
    monkeypatch.setattr("cali._cascade_models.load_cascade_model", verify)
    outputs._verify_model()
    assert "cascade-download chosen-model" in outputs._info.text()
    assert outputs.methods() == ("cascade",)
    verify.side_effect = None
    verify.return_value = SimpleNamespace(
        name="chosen-model",
        sampling_rate=30,
        smoothing=0.025,
        causal_kernel=False,
        noise_levels=(2, 3),
        ensemble_size=5,
        minimum_frames=65,
    )
    outputs._verify_model()
    assert "30 Hz" in outputs._info.text()
    assert "25 ms" in outputs._info.text()
    assert "acausal" in outputs._info.text()
    assert "noise levels 2, 3" in outputs._info.text()
    verify.assert_called_with("chosen-model")
    assert "Rate mismatch: configured acquisition is 10 Hz" in outputs._info.text()
    catalogue = Mock(return_value=())
    monkeypatch.setattr("cali._cascade_models.get_cascade_catalogue", catalogue)
    outputs.set_frame_rate(30)
    assert "matches this model within the allowed 1%" in outputs._info.text()
    assert "checks the recording's acquisition timing" in outputs._info.text()
    catalogue.side_effect = CascadeModelNotFound("Offline catalogue unavailable")
    outputs.set_frame_rate(30.31)
    assert "Rate mismatch" in outputs._info.text()
    assert "30.31 Hz" in outputs._info.text()
    assert "Model list could not be refreshed" in outputs._info.text()
    assert outputs._model.currentText() == "chosen-model"
    assert outputs.methods() == ("cascade",)
    assert verify.call_count == 2  # Editing the rate never re-reads checkpoints.
    outputs._model.setCurrentText("different-model")
    assert "Verified chosen-model" not in outputs._info.text()
    assert outputs._verified_model is None


def test_method_tabs_keep_independent_thresholds_and_full_precision(
    qtbot: QtBot,
) -> None:
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)
    children = tuple(
        SpikeData.from_model(child)
        for child in (
            SpikeAnalysisSettings(
                method="oasis", threshold_value=4.123456, burst_threshold=12.3456
            ),
            SpikeAnalysisSettings(
                method="cascade",
                cascade_ap_threshold_fraction=0.2345678912345678,
                spikes_sync_jitter_window=111.123456,
            ),
        )
    )
    widget.setValue(AnalysisSettingsData(spike_settings=children))
    assert widget.value().spike_settings == children
    model = widget.to_model_settings()
    assert model.get_spike_settings("oasis").threshold_value == 4.123456
    assert (
        model.get_spike_settings("cascade").cascade_ap_threshold_fraction
        == children[1].cascade_ap_threshold_fraction
    )
    widget.set_spike_methods(("oasis",))
    widget.set_spike_methods(("cascade",))
    widget.set_spike_methods(("oasis", "cascade"))
    assert widget.value().spike_settings == children
    threshold = widget._spike_wdg._spike_threshold_wdg._spike_threshold_spin
    threshold.setValue(7)
    assert widget.to_model_settings().get_spike_settings("oasis").threshold_value == 7
    assert (
        widget.to_model_settings().get_spike_settings("cascade").threshold_value is None
    )


def test_cascade_global_requires_explicit_value_and_ap_reset(qtbot: QtBot) -> None:
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)
    widget.set_spike_methods(("cascade",))
    threshold = widget._cascade_spike_wdg._spike_threshold_wdg
    assert threshold.ap_fraction() == 1 / math.e
    threshold._mode.setCurrentIndex(threshold._mode.findData("global"))
    with pytest.raises(ValueError, match="explicit spikes/frame"):
        widget.to_model_settings()
    threshold._global.setText("0.0123456789123")
    settings = widget.to_model_settings().get_spike_settings("cascade")
    assert settings.threshold_value == 0.0123456789123
    assert settings.threshold_mode == "global"
    threshold._global.setText("nan")
    with pytest.raises(ValueError, match="finite"):
        widget.to_model_settings()
    widget.reset()
    settings = widget.to_model_settings().get_spike_settings("cascade")
    assert settings.threshold_mode == "cascade_ap"
    assert settings.cascade_ap_threshold_fraction == 1 / math.e


@pytest.mark.parametrize(
    "method,mode", [("oasis", "cascade_ap"), ("cascade", "multiplier")]
)
def test_gui_rejects_illegal_method_threshold_pair(
    qtbot: QtBot, method: str, mode: str
) -> None:
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)
    with pytest.raises(ValueError, match="Invalid threshold mode"):
        widget.setValue(
            AnalysisSettingsData(
                spike_settings=(SpikeData(method=method, spike_threshold_mode=mode),)
            )
        )


def test_gui_rejects_unknown_legacy_threshold_mode(qtbot: QtBot) -> None:
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)
    with pytest.raises(ValueError, match="threshold_mode"):
        widget.setValue(
            AnalysisSettingsData(spikes_data=SpikeData(spike_threshold_mode="fixed"))
        )


@pytest.mark.parametrize("worker_counts", [None, (3, 4, 2)])
def test_cali_gui_settings_file_restores_children_and_outputs(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_counts: tuple[int, int, int] | None,
) -> None:
    gui = CaliGui(cascade_gui_enabled=True)
    qtbot.addWidget(gui)
    gui._extraction_wdg.setValue(
        ExtractionSettingsData(
            trace_extraction_data=TraceExtractionData(
                spike_methods=("cascade",),
                cascade_model="missing-model",
                cascade_device="mps",
            )
        )
    )
    child = SpikeData.from_model(
        SpikeAnalysisSettings(
            method="cascade", threshold_mode="global", threshold_value=0.123456789
        )
    )
    gui._analysis_wdg.setValue(AnalysisSettingsData(spike_settings=(child,)))
    if worker_counts is not None:
        gui._extraction_wdg._threads.setValue(worker_counts[0])
        gui._analysis_wdg._threads.setValue(worker_counts[1])
        gui._analysis_wdg._n_processes.setValue(worker_counts[2])
    path = tmp_path / "settings.json"
    monkeypatch.setattr(
        "cali.gui._cali_gui.QFileDialog.getSaveFileName", lambda *a: (str(path), "")
    )
    gui._on_save_settings()
    written = json.loads(path.read_text())
    assert written["extraction"]["trace_extraction_data"]["spike_methods"] == [
        "cascade"
    ]
    assert "spike_method" not in written["extraction"]["trace_extraction_data"]
    expected_workers = worker_counts or (1, 1, 1)
    assert (
        written["extraction"]["threads"],
        written["analysis"]["threads"],
        written["analysis"]["n_processes"],
    ) == expected_workers
    if worker_counts is None:
        # Old files can omit worker counts; loading them must replace prior edits.
        written["extraction"].pop("threads")
        written["analysis"].pop("threads")
        written["analysis"].pop("n_processes")
        path.write_text(json.dumps(written))
    gui._extraction_wdg.reset()
    gui._analysis_wdg.reset()
    if worker_counts is None:
        gui._extraction_wdg._threads.setValue(3)
        gui._analysis_wdg._threads.setValue(4)
        gui._analysis_wdg._n_processes.setValue(2)
    errors = Mock()
    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", errors)
    monkeypatch.setattr(
        "cali.gui._cali_gui.QFileDialog.getOpenFileName", lambda *a: (str(path), "")
    )
    gui._on_load_settings()
    errors.assert_not_called()
    assert gui._extraction_wdg.value().trace_extraction_data.cascade_device == "mps"
    assert gui._analysis_wdg.value().spike_settings == (child,)
    assert (
        gui._extraction_wdg.to_model_settings().threads,
        gui._analysis_wdg.to_model_settings().threads,
        gui._analysis_wdg.to_model_settings().n_processes,
    ) == expected_workers


def test_stored_run_loading_and_analysis_only_output_ownership(
    qtbot: QtBot, spike_database: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, path, run_id, methods = spike_database
    with Session(engine) as session:
        owner = session.get(CaliResult, run_id)
        extraction = ExtractionSettings(
            spike_methods=methods,
            cascade_model="cached-model" if "cascade" in methods else None,
            cascade_device="cpu",
            frame_rate=10,
        )
        analysis = AnalysisSettings(
            spike_settings=[SpikeAnalysisSettings(method=method) for method in methods],
            n_processes=2,
        )
        session.add_all([extraction, analysis])
        session.flush()
        extraction_id = extraction.id
        owner.extraction_settings_id, owner.analysis_settings_id = (
            extraction.id,
            analysis.id,
        )
        session.commit()
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._database_path = path
    monkeypatch.setattr(gui, "_on_fov_table_selection_changed", lambda: None)
    errors = Mock()
    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", errors)
    gui._on_run_item_selected(run_id)
    errors.assert_not_called()
    assert gui._extraction_wdg._spike_outputs.methods() == methods
    assert (
        tuple(child.method for child in gui._analysis_wdg.value().spike_settings)
        == methods
    )
    assert gui._analysis_wdg.value().n_processes == 2
    run = gui._run_cali_wdg
    run._extraction_settings_combo.addItem("Select Extraction ID...", None)
    run._extraction_settings_combo.addItem("Stored extraction", extraction_id)
    run._extraction_settings_combo.setCurrentIndex(1)
    run._run_options_combo.setCurrentText(
        "Analysis Only (require detection and extraction)"
    )
    assert not gui._extraction_wdg._spike_outputs._oasis.isEnabled()
    assert not gui._extraction_wdg._spike_outputs._cascade.isEnabled()
    # An editable future selection cannot change analysis-only method ownership.
    gui._extraction_wdg._spike_outputs.setValue(("oasis",), None, "auto")
    assert gui._extraction_wdg._spike_outputs.methods() == methods
    assert (
        tuple(
            child.method
            for child in gui._analysis_wdg.to_model_settings().spike_settings
        )
        == methods
    )


@pytest.mark.parametrize("multi", [False, True])
def test_plot_selector_reads_stored_run_and_dispatches_selected_method(
    qtbot: QtBot, spike_database: tuple, monkeypatch: pytest.MonkeyPatch, multi: bool
) -> None:
    engine, path, run_id, methods = spike_database
    parent = QWidget()
    qtbot.addWidget(parent)
    widget = _MultilWellGraphWidget(parent) if multi else _SingleWellGraphWidget(parent)
    widget.engine, widget.database_path = engine, path
    widget.run_id = run_id
    if not multi:
        widget.fov = "A1_0"
    selected = "cascade" if "cascade" in methods else "oasis"
    assert widget._spike_method == selected
    assert widget._backend_selector.isHidden() == (len(methods) == 1)
    names = [widget._combo.itemText(index) for index in range(widget._combo.count())]
    assert any("CASCADE Expected Spike Rate" in name for name in names) == (
        selected == "cascade"
    )
    assert "Inferred Spikes" in names if not multi else True
    dispatch = Mock()
    name = "plot_multi_well_data" if multi else "plot_single_well_data"
    monkeypatch.setattr(f"cali.gui._pygraph_plot_widgets.{name}", dispatch)
    if not multi:
        widget._combo.setCurrentText("Inferred Spikes")
    else:
        widget._combo.setCurrentIndex(
            next(
                index
                for index in range(widget._combo.count())
                if "Spike" in widget._combo.itemText(index) and index > 0
            )
        )
    assert dispatch.call_args.kwargs["spike_method"] == selected
    if len(methods) == 2:
        selector = widget._backend_selector._combo
        selector.setCurrentIndex(selector.findData("oasis"))
        assert widget._spike_method == "oasis"
        names = [
            widget._combo.itemText(index) for index in range(widget._combo.count())
        ]
        assert not any("CASCADE Expected Spike Rate" in name for name in names)
        if not multi:
            widget._combo.setCurrentText("Inferred Spikes")
            assert dispatch.call_args.kwargs["spike_method"] == "oasis"
