"""Tests for the conditions dialog drag-and-drop functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget

from cali.gui._pygraph_plot_widgets import _ConditionsDialog

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_conditions_dialog_creation(qtbot: QtBot) -> None:
    """Test that conditions dialog is created with correct initial state."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": {"visible": True, "color": "gray"},
        "treatment_A": {"visible": True, "color": "green"},
        "treatment_B": {"visible": False, "color": "magenta"},
        "knockout": {"visible": True, "color": "gray"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Check widget properties
    assert dialog.windowTitle() == "Condition Order, Visibility, and Color"
    assert dialog.isModal()

    # Check list widget has correct number of items
    assert dialog._list_widget.count() == 4

    # Check items are in correct order (check custom widgets)
    item0_widget = dialog._list_widget.itemWidget(dialog._list_widget.item(0))
    assert item0_widget.get_name() == "control"

    item1_widget = dialog._list_widget.itemWidget(dialog._list_widget.item(1))
    assert item1_widget.get_name() == "treatment_A"

    item2_widget = dialog._list_widget.itemWidget(dialog._list_widget.item(2))
    assert item2_widget.get_name() == "treatment_B"

    item3_widget = dialog._list_widget.itemWidget(dialog._list_widget.item(3))
    assert item3_widget.get_name() == "knockout"


def test_conditions_dialog_check_states(qtbot: QtBot) -> None:
    """Test that check states are correctly initialized."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "condition_1": {"visible": True, "color": "gray"},
        "condition_2": {"visible": False, "color": "green"},
        "condition_3": {"visible": True, "color": "magenta"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Check that enabled conditions are checked
    item0 = dialog._list_widget.item(0)
    assert item0 is not None
    widget0 = dialog._list_widget.itemWidget(item0)
    assert widget0.is_visible()

    # Check that disabled conditions are unchecked
    item1 = dialog._list_widget.item(1)
    assert item1 is not None
    widget1 = dialog._list_widget.itemWidget(item1)
    assert not widget1.is_visible()

    item2 = dialog._list_widget.item(2)
    assert item2 is not None
    widget2 = dialog._list_widget.itemWidget(item2)
    assert widget2.is_visible()


def test_conditions_dialog_get_conditions(qtbot: QtBot) -> None:
    """Test that get_conditions returns the correct dictionary."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": {"visible": True, "color": "gray"},
        "treatment": {"visible": False, "color": "green"},
        "knockout": {"visible": True, "color": "magenta"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Get conditions should return the same as input
    result = dialog.get_conditions()
    assert result == conditions
    assert list(result.keys()) == list(conditions.keys())


def test_conditions_dialog_reordering(qtbot: QtBot) -> None:
    """Test that reordering items changes the returned order."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": {"visible": True, "color": "gray"},
        "treatment_A": {"visible": True, "color": "green"},
        "treatment_B": {"visible": False, "color": "magenta"},
        "knockout": {"visible": True, "color": "gray"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Simulate reordering: move first item to position 2
    # Need to preserve the custom widget when reordering
    item = dialog._list_widget.item(0)
    widget = dialog._list_widget.itemWidget(item)

    # Remove from position 0
    dialog._list_widget.takeItem(0)

    # Insert at position 2
    dialog._list_widget.insertItem(2, item)
    dialog._list_widget.setItemWidget(item, widget)

    # Get new order
    result = dialog.get_conditions()

    # Order should be: treatment_A, treatment_B, control, knockout
    expected_order = ["treatment_A", "treatment_B", "control", "knockout"]
    assert list(result.keys()) == expected_order

    # States should be preserved
    assert result["control"]["visible"] is True
    assert result["treatment_A"]["visible"] is True
    assert result["treatment_B"]["visible"] is False
    assert result["knockout"]["visible"] is True


def test_conditions_dialog_toggle_check_state(qtbot: QtBot) -> None:
    """Test that toggling check states is reflected in get_conditions."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "condition_1": {"visible": True, "color": "gray"},
        "condition_2": {"visible": True, "color": "green"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Toggle the check state of the first item
    item0 = dialog._list_widget.item(0)
    assert item0 is not None
    widget0 = dialog._list_widget.itemWidget(item0)
    widget0._checkbox.setChecked(False)

    # Get conditions
    result = dialog.get_conditions()

    # First condition should now be False
    assert result["condition_1"]["visible"] is False
    assert result["condition_2"]["visible"] is True


def test_conditions_dialog_empty_conditions(qtbot: QtBot) -> None:
    """Test dialog with empty conditions dictionary."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {}

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    assert dialog._list_widget.count() == 0
    assert dialog.get_conditions() == {}
