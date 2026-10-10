"""Selected-source timing previews agree with extraction without reading pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from cali._constants import RUNNER_TIME_KEY
from cali.gui import CaliGui
from cali.gui._extraction_gui import _ExtractionGUI

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def timestamps(values: list[float]) -> list[dict]:
    return [
        {RUNNER_TIME_KEY: value, "exposure_ms": 5, "pixel_size_um": 0.5}
        for value in values
    ]


def test_timestamp_cutoff_and_live_settings(qtbot: QtBot) -> None:
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)
    trace = widget._trace_extraction_wdg
    meta = timestamps([1000, 1100, 1250, 1500, 1600, 1700, 1800, 1900])
    widget.set_source_metadata(("A1_0001", len(meta), meta))
    trace._discard_seconds_radio.setChecked(True)
    trace._discard_initial_spin.setValue(0.251)
    text = widget._timing_preview.text()
    assert "acquisition timestamps" in text
    assert "Discard 3 frames; retain 5" in text
    assert "source frame: 4 (1-based)" in text
    assert "Resolved start: 0.5 s" in text
    trace._discard_initial_spin.setValue(9)
    assert "removes the entire recording" in widget._timing_preview.text()
    widget.reset()
    assert "Select a source FOV" in widget._timing_preview.text()


def test_exposure_is_untrusted_until_user_verifies_rate(qtbot: QtBot) -> None:
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)
    trace = widget._trace_extraction_wdg
    widget.set_source_metadata(("Exposure-only TIFF", 100, [{"exposure_ms": 5}]))
    trace._discard_seconds_radio.setChecked(True)
    trace._discard_initial_spin.setValue(0.15)
    assert "exposure only (unverified)" in widget._timing_preview.text()
    assert "Discarding seconds requires" in widget._timing_preview.text()
    widget._metadata_wdg._frame_rate_spin.setValue(20)
    trace._frame_rate_verified.setChecked(True)
    assert "user-verified frame rate" in widget._timing_preview.text()
    assert "Discard 3 frames; retain 97" in widget._timing_preview.text()
    widget._metadata_wdg._frame_rate_spin.setValue(40)
    assert "Discard 6 frames; retain 94" in widget._timing_preview.text()


def test_cascade_checks_retained_intervals_not_discarded_startup(qtbot: QtBot) -> None:
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    widget._metadata_wdg._frame_rate_spin.setValue(10)
    meta = timestamps([0, 200, 300, 400, 500, 600, 700, 800])
    widget.set_source_metadata(("A1", len(meta), meta))
    assert "requires uniform intervals" in widget._timing_preview.text()
    widget._trace_extraction_wdg._discard_initial_spin.setValue(1)
    assert "requires uniform intervals" not in widget._timing_preview.text()
    widget._metadata_wdg._frame_rate_spin.setValue(20)
    assert "does not match extraction settings" in widget._timing_preview.text()


def test_short_and_malformed_recordings_report_reason(qtbot: QtBot) -> None:
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)
    widget.set_source_metadata(("Short TIFF", 4, [{"frame_period_ms": 100}]))
    assert "requires at least 5 frames" in widget._timing_preview.text()
    widget.set_source_metadata(("Broken timestamps", 2, timestamps([1, 1])))
    assert "strictly increasing" in widget._timing_preview.text()
    widget.set_source_metadata(None, error="Source recording is unavailable")
    assert "Source recording is unavailable" in widget._timing_preview.text()


@pytest.mark.parametrize("analysis", [False, True])
@pytest.mark.parametrize("source", ["timestamps", "period", "exposure"])
def test_metadata_loader_uses_selected_source_and_never_inverse_exposure(
    qtbot: QtBot, analysis: bool, source: str
) -> None:
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._data = MagicMock()
    meta = timestamps([1000 + 50 * t for t in range(8)])
    if source != "timestamps":
        meta = [{"exposure_ms": 5, "pixel_size_um": 0.5} for _ in range(8)]
    if source == "period":
        for item in meta:
            item["frame_period_ms"] = 50
    gui._data.position_metadata.return_value = (8, meta)
    gui._extraction_wdg._metadata_wdg._frame_rate_spin.setValue(30)
    gui._extraction_wdg._trace_extraction_wdg._frame_rate_verified.setChecked(True)
    value = MagicMock(pos_idx=2)
    value.fov.name = "B2_0001"
    with (
        patch.object(gui._fov_table, "selectedItems", return_value=[object()]),
        patch.object(gui._fov_table, "value", return_value=value),
        patch("cali.gui._cali_gui.show_error_dialog") as dialog,
    ):
        if analysis:
            gui._on_analysis_meta_clicked()
        else:
            gui._on_extraction_meta_clicked()
    gui._data.position_metadata.assert_called_once_with(2)
    gui._data.isel.assert_not_called()
    expected = 30 if source == "exposure" else 20
    assert gui._extraction_wdg._metadata_wdg.value().frame_rate == expected
    assert gui._analysis_wdg._metadata_wdg.value() == expected
    assert (
        not gui._extraction_wdg._trace_extraction_wdg._frame_rate_verified.isChecked()
    )
    assert "B2_0001" in gui._extraction_wdg._timing_preview.text()
    if source == "exposure":
        assert "Exposure duration alone" in dialog.call_args.args[1]
    else:
        dialog.assert_not_called()


def test_fov_selection_and_database_only_clear_source_preview(qtbot: QtBot) -> None:
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._extraction_wdg.set_source_metadata(
        ("Previous", 10, [{"frame_period_ms": 100}])
    )
    with patch.object(gui._fov_table, "selectedItems", return_value=[]):
        gui._on_fov_table_selection_changed()
    assert "Previous" not in gui._extraction_wdg._timing_preview.text()
    with patch("cali.gui._cali_gui.show_error_dialog") as dialog:
        gui._on_extraction_meta_clicked()
    assert "Data not loaded" in dialog.call_args.args[1]
