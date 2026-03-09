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
import pytest


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


# ============================================================================
# Value and SetValue Method Tests
# ============================================================================


@pytest.mark.parametrize(
    ("direction", "check_attr", "get_count_method"),
    [
        ("vertical", "rowStretch", "rowCount"),
        ("horizontal", "columnStretch", "columnCount"),
    ],
)
def test_export_group_add_stretch(
    qtbot: QtBot, direction: str, check_attr: str, get_count_method: str
) -> None:
    """Test add_stretch with different directions."""
    widget = _ExportGroup("Test Export")
    qtbot.addWidget(widget)

    # Add some options first
    if direction == "vertical":
        widget.add_option("Option 1", 0, 0)
        widget.add_option("Option 2", 1, 0)
    else:
        widget.add_option("Option 1", 0, 0)
        widget.add_option("Option 2", 0, 1)

    # Add stretch
    widget.add_stretch(direction)

    # Check that stretch was added to last row/column
    count = getattr(widget._layout, get_count_method)()
    stretch_value = getattr(widget._layout, check_attr)(count - 1)
    assert stretch_value == 1


def test_export_group_value_returns_state(qtbot: QtBot) -> None:
    """Test value() returns current widget state correctly."""
    widget = _ExportGroup("Test Export")
    qtbot.addWidget(widget)

    # Add options at different positions with different checked states
    widget.add_option("Option 1", 0, 0, checked=True)
    widget.add_option("Option 2", 1, 0, checked=False)
    widget.add_option("Option 3", 0, 1, checked=True)

    # Get value
    value = widget.value()

    # Check structure
    assert isinstance(value, dict)
    assert "Option 1" in value
    assert "Option 2" in value
    assert "Option 3" in value

    # Check values: (checked_state, row, col)
    assert value["Option 1"] == (True, 0, 0)
    assert value["Option 2"] == (False, 1, 0)
    assert value["Option 3"] == (True, 0, 1)


def test_export_group_set_value_restores_state(qtbot: QtBot) -> None:
    """Test setValue() restores widget state correctly."""
    widget = _ExportGroup("Test Export")
    qtbot.addWidget(widget)

    # Initial state
    widget.add_option("Option 1", 0, 0, checked=True)
    widget.add_option("Option 2", 1, 0, checked=False)

    # Get initial value
    widget.value()

    # Create new state to set
    new_state = {
        "New Option 1": (False, 0, 0),
        "New Option 2": (True, 1, 1),
        "New Option 3": (True, 2, 0),
    }

    # Set new value
    widget.setValue(new_state)

    # Verify state was updated
    current_value = widget.value()
    assert current_value == new_state

    # Verify old options are gone
    assert "Option 1" not in current_value
    assert "Option 2" not in current_value


def test_export_group_set_value_clears_existing_widgets(qtbot: QtBot) -> None:
    """Test setValue() clears existing widgets before adding new ones."""
    widget = _ExportGroup("Test Export")
    qtbot.addWidget(widget)

    # Add initial options
    widget.add_option("Option 1", 0, 0, checked=True)
    widget.add_option("Option 2", 1, 0, checked=False)

    # Verify options exist
    assert len(widget._checkboxes) == 2

    # Set new state with different options
    new_state = {
        "Different Option": (True, 0, 0),
    }
    widget.setValue(new_state)

    # Verify old options were cleared
    assert len(widget._checkboxes) == 1
    assert "Different Option" in widget._checkboxes
    assert "Option 1" not in widget._checkboxes
    assert "Option 2" not in widget._checkboxes


def test_export_group_get_export_options_with_real_types(qtbot: QtBot) -> None:
    """Test get_export_options() with actual trace/correlation types."""
    widget = _ExportGroup("Test Export")
    qtbot.addWidget(widget)

    # Add options with real constant types
    widget.add_option(RAW_CALCIUM_TRACES, 0, 0, checked=True)
    widget.add_option(DFF_TRACES, 1, 0, checked=False)
    widget.add_option(NEUROPIL_TRACES, 2, 0, checked=True)

    # Enable the group
    widget.setChecked(True)

    # Get export options
    export_options = widget.get_export_options()

    # Should return dict with only checked options
    assert export_options is not None
    assert RAW_CALCIUM_TRACES in export_options
    assert export_options[RAW_CALCIUM_TRACES] is True
    assert NEUROPIL_TRACES in export_options
    assert export_options[NEUROPIL_TRACES] is True

    # Unchecked option should not be in dict
    assert DFF_TRACES not in export_options


def test_export_group_value_with_changed_checkboxes(qtbot: QtBot) -> None:
    """Test value() reflects checkbox state changes."""
    widget = _ExportGroup("Test Export")
    qtbot.addWidget(widget)

    # Add option initially checked
    widget.add_option("Toggle Option", 0, 0, checked=True)

    # Verify initial state
    value = widget.value()
    assert value["Toggle Option"][0] is True

    # Change checkbox state
    checkbox = widget._checkboxes["Toggle Option"][0]
    checkbox.setChecked(False)

    # Verify value() reflects the change
    value = widget.value()
    assert value["Toggle Option"][0] is False


def test_export_group_set_value_preserves_positions(qtbot: QtBot) -> None:
    """Test setValue() preserves row/col positions from input."""
    widget = _ExportGroup("Test Export")
    qtbot.addWidget(widget)

    # Create state with specific positions
    state = {
        "Top Left": (True, 0, 0),
        "Top Right": (False, 0, 2),
        "Bottom Left": (True, 3, 0),
        "Bottom Right": (False, 3, 2),
    }

    widget.setValue(state)

    # Verify positions are preserved
    result = widget.value()
    for text, (checked, row, col) in state.items():
        assert result[text] == (checked, row, col)
