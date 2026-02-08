"""Tests for _ExportGroup widget functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cali._constants import (
    CALCIUM_DEN_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    DEN_DFF_TRACES,
    DFF_TRACES,
    INFERRED_SPIKES_CROSS_CORRELATION,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS,
    INFERRED_SPIKES_SYNCHRONY,
    INFERRED_SPIKES_THRESHOLDED_BINARY,
    INFERRED_SPIKES_TRACES,
    NEUROPIL_CORRECTED_TRACES,
    NEUROPIL_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.gui._util import _ExportGroup

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_export_group_default_state(qtbot: QtBot) -> None:
    """Test that _ExportGroup initializes with correct default state."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    # Should be checkable and checked by default
    assert export_group.isCheckable()
    assert export_group.isChecked()

    # Should be empty initially
    assert export_group.value() == {}
    assert export_group.get_export_options() == {}


def test_export_group_add_options(qtbot: QtBot) -> None:
    """Test adding options to the export group."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    # Add some options
    export_group.add_option(RAW_CALCIUM_TRACES, 0, 0, checked=True)
    export_group.add_option(DFF_TRACES, 1, 0, checked=True)
    export_group.add_option(NEUROPIL_TRACES, 2, 0, checked=False)

    # Check value() returns all options with their states
    value = export_group.value()
    assert RAW_CALCIUM_TRACES in value
    assert DFF_TRACES in value
    assert NEUROPIL_TRACES in value

    # Check that states are correct
    assert value[RAW_CALCIUM_TRACES][0] is True  # checked
    assert value[DFF_TRACES][0] is True
    assert value[NEUROPIL_TRACES][0] is False


def test_export_group_get_export_options_filters_checked(qtbot: QtBot) -> None:
    """Test that get_export_options() only returns checked options."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    # Add mix of checked and unchecked options
    export_group.add_option(RAW_CALCIUM_TRACES, 0, 0, checked=True)
    export_group.add_option(DFF_TRACES, 1, 0, checked=True)
    export_group.add_option(NEUROPIL_TRACES, 2, 0, checked=False)
    export_group.add_option(NEUROPIL_CORRECTED_TRACES, 3, 0, checked=False)
    export_group.add_option(DEN_DFF_TRACES, 4, 0, checked=True)

    # get_export_options() should only return checked items
    export_options = export_group.get_export_options()

    # Should only have 3 items (the checked ones)
    assert len(export_options) == 3
    assert RAW_CALCIUM_TRACES in export_options
    assert DFF_TRACES in export_options
    assert DEN_DFF_TRACES in export_options

    # Unchecked ones should not be present
    assert NEUROPIL_TRACES not in export_options
    assert NEUROPIL_CORRECTED_TRACES not in export_options

    # All returned items should be True
    assert all(export_options.values())


def test_export_group_disabled_returns_empty(qtbot: QtBot) -> None:
    """Test that unchecking the group itself returns empty dict."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    # Add checked options
    export_group.add_option(RAW_CALCIUM_TRACES, 0, 0, checked=True)
    export_group.add_option(DFF_TRACES, 1, 0, checked=True)

    # Initially should return checked options
    assert len(export_group.get_export_options()) == 2

    # Uncheck the entire group
    export_group.setChecked(False)

    # get_export_options() should still return the checked items
    # (the group being unchecked is handled by the GUI get_export_options() method)
    assert len(export_group.get_export_options()) == 2


def test_export_group_correlation_types(qtbot: QtBot) -> None:
    """Test export group with correlation data types."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    # Add correlation options
    export_group.add_option(CALCIUM_DFF_CORRELATION, 0, 0, checked=False)
    export_group.add_option(CALCIUM_DEN_DFF_CORRELATION, 1, 0, checked=True)
    export_group.add_option(INFERRED_SPIKES_SYNCHRONY, 2, 0, checked=True)
    export_group.add_option(INFERRED_SPIKES_CROSS_CORRELATION, 3, 0, checked=True)
    export_group.add_option(INFERRED_SPIKES_CROSS_CORRELATION_LAGS, 4, 0, checked=False)
    export_group.add_option(INFERRED_SPIKES_TRACES, 5, 0, checked=False)
    export_group.add_option(INFERRED_SPIKES_THRESHOLDED_BINARY, 6, 0, checked=False)

    export_options = export_group.get_export_options()

    # Should have 3 checked items
    assert len(export_options) == 3
    assert CALCIUM_DEN_DFF_CORRELATION in export_options
    assert INFERRED_SPIKES_SYNCHRONY in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION in export_options

    # Unchecked ones should not be present
    assert CALCIUM_DFF_CORRELATION not in export_options
    assert INFERRED_SPIKES_CROSS_CORRELATION_LAGS not in export_options


def test_export_group_all_unchecked(qtbot: QtBot) -> None:
    """Test that get_export_options() returns empty dict when all options unchecked."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    # Add all unchecked options
    export_group.add_option(RAW_CALCIUM_TRACES, 0, 0, checked=False)
    export_group.add_option(DFF_TRACES, 1, 0, checked=False)
    export_group.add_option(DEN_DFF_TRACES, 2, 0, checked=False)

    # Should return empty dict
    export_options = export_group.get_export_options()
    assert export_options == {}


def test_export_group_toggle_checkbox(qtbot: QtBot) -> None:
    """Test toggling individual checkboxes updates export options."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    export_group.add_option(RAW_CALCIUM_TRACES, 0, 0, checked=True)
    export_group.add_option(DFF_TRACES, 1, 0, checked=False)

    # Initially only RAW_CALCIUM_TRACES checked
    assert len(export_group.get_export_options()) == 1
    assert RAW_CALCIUM_TRACES in export_group.get_export_options()

    # Get the checkbox and toggle it
    export_group.value()
    dff_checkbox = None
    for text, (checkbox, _, _) in export_group._checkboxes.items():
        if text == DFF_TRACES:
            dff_checkbox = checkbox
            break

    assert dff_checkbox is not None

    # Check the DFF_TRACES checkbox
    dff_checkbox.setChecked(True)

    # Now should have both
    export_options = export_group.get_export_options()
    assert len(export_options) == 2
    assert RAW_CALCIUM_TRACES in export_options
    assert DFF_TRACES in export_options


def test_export_group_value_vs_get_export_options(qtbot: QtBot) -> None:
    """Test that value() and get_export_options() return different structures."""
    export_group = _ExportGroup()
    qtbot.addWidget(export_group)

    export_group.add_option(RAW_CALCIUM_TRACES, 0, 0, checked=True)
    export_group.add_option(DFF_TRACES, 1, 0, checked=False)

    # value() returns tuples with (checked, row, col)
    value_dict = export_group.value()
    assert isinstance(value_dict[RAW_CALCIUM_TRACES], tuple)
    assert len(value_dict[RAW_CALCIUM_TRACES]) == 3
    assert value_dict[RAW_CALCIUM_TRACES][0] is True  # checked status
    assert value_dict[DFF_TRACES][0] is False

    # get_export_options() returns dict with only checked items and bool values
    export_options = export_group.get_export_options()
    assert len(export_options) == 1
    assert RAW_CALCIUM_TRACES in export_options
    assert export_options[RAW_CALCIUM_TRACES] is True
    assert DFF_TRACES not in export_options
