"""Tests for GUI export options integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cali._constants import (
    CALCIUM_DEC_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    DFF_TRACES,
    INFERRED_SPIKES_CROSS_CORRELATION,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS,
    INFERRED_SPIKES_SYNCHRONY,
    NEUROPIL_CORRECTED_TRACES,
    NEUROPIL_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.gui._analysis_gui import _AnalysisGUI
from cali.gui._extraction_gui import _ExtractionGUI

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


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

    # By default, CALCIUM_DFF_CORRELATION is unchecked (row 0, checked=False)
    # Other correlations are checked by default

    # Unchecked option should NOT be in export_options
    assert CALCIUM_DFF_CORRELATION not in export_options

    # Checked options should be in export_options
    assert CALCIUM_DEC_DFF_CORRELATION in export_options
    assert INFERRED_SPIKES_SYNCHRONY in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION_LAGS in export_options


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

    export_options = widget.get_export_options()
    assert export_options is not None

    # Every value in the dict should be True (checked)
    assert all(export_options.values())

    # Unchecked items should not be present at all (not even with False value)
    for key, value in export_options.items():
        assert value is True, f"{key} should be True, not False"
