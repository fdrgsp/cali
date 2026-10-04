"""Settings selections preserve computation ownership and keyboard access."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QStyle, QStyleOptionGroupBox

from cali.gui import CaliGui
from cali.gui._analysis_gui import _AnalysisGUI
from cali.gui._extraction_gui import _ExtractionGUI
from cali.gui._run_widget import _RunCaliWidget

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_extraction_groups_keep_both_outputs_and_parameters(qtbot: QtBot) -> None:
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    widget.resize(700, 750)
    widget.show()
    widget._settings_tabs.setCurrentIndex(1)
    outputs = widget._spike_outputs
    outputs._model.setCurrentText("explicit-model")
    outputs._device.setCurrentIndex(outputs._device.findData("mps"))
    assert outputs.methods() == ("cascade",)
    assert outputs._cascade.isCheckable() and outputs._cascade.isVisible()
    assert outputs._oasis.isCheckable() and outputs._oasis.isVisible()
    decay = widget._trace_extraction_wdg._decay_constant_spin
    assert decay.isEnabled() and decay.isVisible()
    decay.setValue(1.5)
    option = QStyleOptionGroupBox()
    outputs._oasis.initStyleOption(option)
    check_position = (
        outputs._oasis.style()
        .subControlRect(
            QStyle.ComplexControl.CC_GroupBox,
            option,
            QStyle.SubControl.SC_GroupBoxCheckBox,
            outputs._oasis,
        )
        .center()
    )
    qtbot.mouseClick(outputs._oasis, Qt.MouseButton.LeftButton, pos=check_position)
    assert outputs.methods() == ("oasis", "cascade")
    model = widget.to_model_settings()
    assert model.spike_methods == ("oasis", "cascade")
    assert model.cascade_model == "explicit-model"
    assert model.cascade_device == "mps"
    assert model.decay_constant == 1.5


def test_keyboard_cannot_uncheck_last_spike_output(qtbot: QtBot) -> None:
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    widget.show()
    widget._settings_tabs.setCurrentIndex(1)
    outputs = widget._spike_outputs
    outputs._cascade.setFocus()
    qtbot.keyClick(outputs._cascade, Qt.Key.Key_Space)
    assert outputs.methods() == ("cascade",)
    qtbot.keyClick(outputs._oasis, Qt.Key.Key_Space)
    assert outputs.methods() == ("oasis", "cascade")
    qtbot.keyClick(outputs._cascade, Qt.Key.Key_Space)
    assert outputs.methods() == ("oasis",)
    qtbot.keyClick(outputs._oasis, Qt.Key.Key_Space)
    assert outputs.methods() == ("oasis",)
    qtbot.keyClick(outputs._cascade, Qt.Key.Key_Space)
    qtbot.keyClick(outputs._oasis, Qt.Key.Key_Space)
    assert outputs.methods() == ("cascade",)
    assert widget._trace_extraction_wdg._decay_constant_spin.isEnabled()
    widget.setEnabled(False)
    assert not widget._trace_extraction_wdg._decay_constant_spin.isEnabled()
    widget.setEnabled(True)
    assert widget._trace_extraction_wdg._decay_constant_spin.isEnabled()


def test_gated_cascade_group_explains_unavailable_output(qtbot: QtBot) -> None:
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)
    widget.show()
    widget._settings_tabs.setCurrentIndex(1)
    outputs = widget._spike_outputs
    assert outputs._status.isVisible()
    assert "release" in outputs._status.text()
    assert not outputs._cascade.isEnabled()
    assert not outputs._model.isEnabled()
    assert widget.to_model_settings().spike_methods == ("oasis",)


def test_analysis_tab_checks_require_one_pillar(qtbot: QtBot) -> None:
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)
    widget.show()
    widget.set_spike_methods(("oasis", "cascade"))
    qtbot.mouseClick(widget._enable_calcium_cb, Qt.MouseButton.LeftButton)
    assert not widget.to_model_settings().enable_calcium
    assert not widget._calcium_peaks_wdg.isEnabled()
    qtbot.keyClick(widget._enable_spikes_cb, Qt.Key.Key_Space)
    assert widget.to_model_settings().enable_spikes
    assert tuple(
        child.method for child in widget.to_model_settings().spike_settings
    ) == ("oasis", "cascade")
    qtbot.mouseClick(widget._enable_calcium_cb, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(widget._enable_spikes_cb, Qt.MouseButton.LeftButton)
    assert not widget.to_model_settings().enable_spikes
    assert not widget._spike_tabs.isEnabled()


def test_run_controls_explain_reanalysis_and_export(qtbot: QtBot) -> None:
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)
    widget._run_options_combo.setCurrentIndex(5)
    assert "saved traces" in widget._run_help.text()
    assert "re-extraction" in widget._run_help.text()
    widget._run_options_combo.setCurrentIndex(6)
    assert "No new calculations" in widget._run_help.text()
    assert widget._save_settings_btn.text() == "Save settings"
    assert widget._load_settings_btn.text() == "Load settings"


def test_experiment_header_opens_dialog_and_tracks_available_data(qtbot: QtBot) -> None:
    with patch.object(CaliGui, "_show_data_input_dialog") as open_dialog:
        gui = CaliGui()
        qtbot.addWidget(gui)
        gui.show()
        assert "to begin" in gui._experiment_context.text()
        qtbot.mouseClick(gui._open_experiment_btn, Qt.MouseButton.LeftButton)
        open_dialog.assert_called_once()
        gui._enable(False)
        assert not gui._open_experiment_btn.isEnabled()
        gui._enable(True)
        assert gui._open_experiment_btn.isEnabled()
        gui._database_path = "/tmp/example.cali"
        gui._refresh_experiment_context()
        assert "saved results only" in gui._experiment_context.text()
        assert "need imaging data" in gui._experiment_context.text()
        gui._data = MagicMock()
        gui._refresh_experiment_context()
        assert "imaging data loaded" in gui._experiment_context.text()
        gui._data = None
        gui._database_path = None
        gui._refresh_experiment_context()
        assert "to begin" in gui._experiment_context.text()
