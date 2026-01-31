"""Tests for _ExportGroup value() and setValue() methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cali._constants import DFF_TRACES, NEUROPIL_TRACES, RAW_CALCIUM_TRACES
from cali.gui._util import _ExportGroup

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


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
