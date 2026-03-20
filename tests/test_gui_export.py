"""Tests for GUI export options integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cali._constants import (
    CALCIUM_DEN_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    DFF_TRACES,
    INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES,
    INFERRED_SPIKES_SYNCHRONY,
    INFERRED_SPIKES_SYNCHRONY_RISING_EDGES,
    INFERRED_SPIKES_THRESHOLDED_BINARY,
    MULTI_WELL_AGGREGATED_DATA,
    NEUROPIL_CORRECTED_TRACES,
    NEUROPIL_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.gui._analysis_gui import _AnalysisGUI
from cali.gui._extraction_gui import _ExtractionGUI

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot
from pathlib import Path
from unittest.mock import patch

import pytest

from cali.gui._cali_gui import CaliGui


def test_extraction_gui_get_export_options_filters_unchecked(qtbot: QtBot) -> None:
    """Test that _ExtractionGUI.get_export_options() only returns checked options."""
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)

    # By default, the export group should be enabled
    assert widget._export_group.isChecked()

    # Get export options - should only include checked items
    export_options = widget.get_export_options()
    assert export_options is not None

    # By default, some options are checked and some are not
    # RAW_CALCIUM_TRACES should be checked (row 0)
    # NEUROPIL_TRACES should be unchecked (row 1, checked=False)
    # NEUROPIL_CORRECTED_TRACES should be unchecked (row 2, checked=False)
    # DFF_TRACES should be checked (row 3)

    # Unchecked options should NOT be in export_options
    assert NEUROPIL_TRACES not in export_options
    assert NEUROPIL_CORRECTED_TRACES not in export_options

    # Checked options should be in export_options
    assert RAW_CALCIUM_TRACES in export_options
    assert DFF_TRACES in export_options


def test_extraction_gui_export_disabled(qtbot: QtBot) -> None:
    """Test that unchecking the export group returns None."""
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)

    # Disable the entire export group
    widget._export_group.setChecked(False)

    # Should return None when group is disabled
    assert widget.get_export_options() is None


def test_analysis_gui_get_export_options_filters_unchecked(qtbot: QtBot) -> None:
    """Test that _AnalysisGUI.get_export_options() only returns checked options."""
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # By default, the export group should be enabled
    assert widget._export_group.isChecked()

    # Get export options
    export_options = widget.get_export_options()
    assert export_options is not None

    # By default, only MULTI_WELL_AGGREGATED_DATA is checked;
    # all correlation options are unchecked
    assert MULTI_WELL_AGGREGATED_DATA in export_options

    # Unchecked options should NOT be in export_options
    assert CALCIUM_DFF_CORRELATION not in export_options
    assert CALCIUM_DEN_DFF_CORRELATION not in export_options
    assert INFERRED_SPIKES_THRESHOLDED_BINARY not in export_options
    assert INFERRED_SPIKES_SYNCHRONY not in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION not in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION_LAGS not in export_options


def test_analysis_gui_export_disabled(qtbot: QtBot) -> None:
    """Test that unchecking the export group returns None."""
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Disable the entire export group
    widget._export_group.setChecked(False)

    # Should return None when group is disabled
    assert widget.get_export_options() is None


def test_extraction_gui_toggle_individual_options(qtbot: QtBot) -> None:
    """Test toggling individual checkboxes in extraction GUI."""
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)

    # Get initial export options
    initial_options = widget.get_export_options()
    assert initial_options is not None
    assert NEUROPIL_TRACES not in initial_options

    # Find and check the NEUROPIL_TRACES checkbox
    for text, (checkbox, _, _) in widget._export_group._checkboxes.items():
        if text == NEUROPIL_TRACES:
            checkbox.setChecked(True)
            break

    # Now it should be in export options
    updated_options = widget.get_export_options()
    assert updated_options is not None
    assert NEUROPIL_TRACES in updated_options
    assert updated_options[NEUROPIL_TRACES] is True


def test_analysis_gui_toggle_individual_options(qtbot: QtBot) -> None:
    """Test toggling individual checkboxes in analysis GUI."""
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Enable export group for testing
    widget._export_group.setChecked(True)

    # Initially CALCIUM_DFF_CORRELATION should not be in options
    initial_options = widget.get_export_options()
    assert initial_options is not None
    assert CALCIUM_DFF_CORRELATION not in initial_options

    # Find and check the CALCIUM_DFF_CORRELATION checkbox
    for text, (checkbox, _, _) in widget._export_group._checkboxes.items():
        if text == CALCIUM_DFF_CORRELATION:
            checkbox.setChecked(True)
            break

    # Now it should be in export options
    updated_options = widget.get_export_options()
    assert updated_options is not None
    assert CALCIUM_DFF_CORRELATION in updated_options
    assert updated_options[CALCIUM_DFF_CORRELATION] is True


def test_extraction_gui_no_unchecked_in_export_dict(qtbot: QtBot) -> None:
    """Test that unchecked items are completely absent from export options."""
    widget = _ExtractionGUI()
    qtbot.addWidget(widget)

    export_options = widget.get_export_options()
    assert export_options is not None

    # Every value in the dict should be True (checked)
    assert all(export_options.values())

    # Unchecked items should not be present at all (not even with False value)
    for key, value in export_options.items():
        assert value is True, f"{key} should be True, not False"


def test_analysis_gui_no_unchecked_in_export_dict(qtbot: QtBot) -> None:
    """Test that unchecked items are completely absent from export options."""
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Enable export group for testing
    widget._export_group.setChecked(True)

    export_options = widget.get_export_options()
    assert export_options is not None

    # Every value in the dict should be True (checked)
    assert all(export_options.values())

    # Unchecked items should not be present at all (not even with False value)
    for key, value in export_options.items():
        assert value is True, f"{key} should be True, not False"


def test_analysis_gui_reset(qtbot: QtBot) -> None:
    """Test that AnalysisGUI.reset() resets all widgets to defaults."""
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Change some values
    widget._threads.setValue(10)

    # Reset
    widget.reset()

    # Check that thread count was reset to default (cpu_count - 2, min 1)
    import os

    expected_threads = max((os.cpu_count() or 1) - 2, 1)
    assert widget._threads.value() == expected_threads


def test_analysis_gui_rising_edge_options_false_by_default(qtbot: QtBot) -> None:
    """Test that rising edge export options are unchecked by default."""
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Enable export group for testing
    widget._export_group.setChecked(True)

    # Get export options
    export_options = widget.get_export_options()
    assert export_options is not None

    # All rising edge options should NOT be in the export dict by default
    # (because they are unchecked)
    assert INFERRED_SPIKES_SYNCHRONY_RISING_EDGES not in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES not in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES not in export_options
    assert INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES not in export_options


def test_analysis_gui_enable_rising_edge_exports(qtbot: QtBot) -> None:
    """Test that rising edge exports can be enabled individually."""
    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Enable export group for testing
    widget._export_group.setChecked(True)

    # Find and check the rising edge synchrony checkbox
    for text, (checkbox, _, _) in widget._export_group._checkboxes.items():
        if text == INFERRED_SPIKES_SYNCHRONY_RISING_EDGES:
            checkbox.setChecked(True)
            break

    # Now it should be in export options
    export_options = widget.get_export_options()
    assert export_options is not None
    assert INFERRED_SPIKES_SYNCHRONY_RISING_EDGES in export_options
    assert export_options[INFERRED_SPIKES_SYNCHRONY_RISING_EDGES] is True

    # Other rising edge options should still be unchecked
    assert INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES not in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES not in export_options
    assert INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES not in export_options


# ============================================================================
# CaliGui Export-Only Handler Tests
# ============================================================================


@pytest.fixture
def test_data_path() -> Path:
    """Return path to test data directory."""
    return Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")


def test_handle_export_only_no_database(qtbot: QtBot) -> None:
    """Test _handle_export_only with no database loaded."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Explicitly set database_path to None
    gui._database_path = None

    # Mock show_error_dialog
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_error:
        gui._handle_export_only(run_id=1)
        mock_error.assert_called_once()
        assert "No database loaded" in mock_error.call_args[0][1]


def test_handle_export_only_no_exports_selected(
    qtbot: QtBot,
    test_db_copy: Path,
    test_data_path: Path,
) -> None:
    """Test _handle_export_only with no export options selected."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Initialize GUI with database
    gui._initialize_from_database(str(test_db_copy), str(test_data_path))
    qtbot.wait(100)

    # Disable all export options
    gui._extraction_wdg._export_group.setChecked(False)
    gui._analysis_wdg._export_group.setChecked(False)

    # Mock show_error_dialog
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_error:
        gui._handle_export_only(run_id=1)
        mock_error.assert_called_once()
        assert "No export options selected" in mock_error.call_args[0][1]


def test_handle_export_only_export_failure(
    qtbot: QtBot,
    test_db_copy: Path,
    test_data_path: Path,
) -> None:
    """Test _handle_export_only handles export failure gracefully."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Initialize GUI with database
    gui._initialize_from_database(str(test_db_copy), str(test_data_path))
    qtbot.wait(100)

    # Enable some export options
    gui._extraction_wdg._export_group.setChecked(True)

    # Mock export function to raise exception
    with (
        patch("cali.gui._cali_gui.show_error_dialog") as mock_error,
        patch(
            "cali.util._database_to_csv.export_traces_to_csv",
            side_effect=RuntimeError("Test error"),
        ),
    ):
        gui._handle_export_only(run_id=1)
        # Should show error dialog
        mock_error.assert_called()
        assert "Export failed" in mock_error.call_args[0][1]


def test_get_run_option_export_only(qtbot: QtBot) -> None:
    """Test _get_run_option returns correct value for Export Only mode."""
    from cali.gui._run_widget import CaliRunSettings

    gui = CaliGui()
    qtbot.addWidget(gui)

    # Create settings with run_id (Export Only mode)
    settings = CaliRunSettings(
        positions=[],
        run_detection=False,
        run_extraction=False,
        run_analysis=False,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=1,  # This indicates Export Only mode
    )

    result = gui._get_run_option(settings)
    assert result == 6  # Export Only option index


def test_get_run_option_detection_extraction_analysis(qtbot: QtBot) -> None:
    """Test _get_run_option returns correct value for full pipeline."""
    from cali.gui._run_widget import CaliRunSettings

    gui = CaliGui()
    qtbot.addWidget(gui)

    # Create settings for full pipeline
    settings = CaliRunSettings(
        positions=[],
        run_detection=True,
        run_extraction=True,
        run_analysis=True,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=None,
    )

    result = gui._get_run_option(settings)
    assert result == 0  # Detection, Extraction and Analysis


def test_get_run_option_detection_extraction(qtbot: QtBot) -> None:
    """Test _get_run_option returns correct value for detection + extraction."""
    from cali.gui._run_widget import CaliRunSettings

    gui = CaliGui()
    qtbot.addWidget(gui)

    settings = CaliRunSettings(
        positions=[],
        run_detection=True,
        run_extraction=True,
        run_analysis=False,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=None,
    )

    result = gui._get_run_option(settings)
    assert result == 1  # Detection and Extraction


def test_on_scene_well_changed_no_data(qtbot: QtBot) -> None:
    """Test _on_scene_well_changed when no data is loaded."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Should handle gracefully when no data
    gui._on_scene_well_changed()
    # No error should be raised
