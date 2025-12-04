"""Tests for the conditions dialog drag-and-drop functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

from cali.gui._pygraph_plot_widgets import _ConditionsDialog

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_conditions_dialog_creation(qtbot: QtBot) -> None:
    """Test that conditions dialog is created with correct initial state."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": True,
        "treatment_A": True,
        "treatment_B": False,
        "knockout": True,
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Check widget properties
    assert dialog.windowTitle() == "Condition Order and Visibility"
    assert dialog.isModal()

    # Check list widget has correct number of items
    assert dialog._list_widget.count() == 4

    # Check items are in correct order
    assert dialog._list_widget.item(0).text() == "control"
    assert dialog._list_widget.item(1).text() == "treatment_A"
    assert dialog._list_widget.item(2).text() == "treatment_B"
    assert dialog._list_widget.item(3).text() == "knockout"


def test_conditions_dialog_check_states(qtbot: QtBot) -> None:
    """Test that check states are correctly initialized."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "condition_1": True,
        "condition_2": False,
        "condition_3": True,
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Check that enabled conditions are checked
    item0 = dialog._list_widget.item(0)
    assert item0 is not None
    assert item0.checkState() == Qt.CheckState.Checked

    # Check that disabled conditions are unchecked
    item1 = dialog._list_widget.item(1)
    assert item1 is not None
    assert item1.checkState() == Qt.CheckState.Unchecked

    item2 = dialog._list_widget.item(2)
    assert item2 is not None
    assert item2.checkState() == Qt.CheckState.Checked


def test_conditions_dialog_get_conditions(qtbot: QtBot) -> None:
    """Test that get_conditions returns the correct dictionary."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": True,
        "treatment": False,
        "knockout": True,
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
        "control": True,
        "treatment_A": True,
        "treatment_B": False,
        "knockout": True,
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Simulate reordering: move first item to position 2
    item = dialog._list_widget.takeItem(0)
    dialog._list_widget.insertItem(2, item)

    # Get new order
    result = dialog.get_conditions()

    # Order should be: treatment_A, treatment_B, control, knockout
    expected_order = ["treatment_A", "treatment_B", "control", "knockout"]
    assert list(result.keys()) == expected_order

    # States should be preserved
    assert result["control"] is True
    assert result["treatment_A"] is True
    assert result["treatment_B"] is False
    assert result["knockout"] is True


def test_conditions_dialog_toggle_check_state(qtbot: QtBot) -> None:
    """Test that toggling check states is reflected in get_conditions."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "condition_1": True,
        "condition_2": True,
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Toggle the check state of the first item
    item0 = dialog._list_widget.item(0)
    assert item0 is not None
    item0.setCheckState(Qt.CheckState.Unchecked)

    # Get conditions
    result = dialog.get_conditions()

    # First condition should now be False
    assert result["condition_1"] is False
    assert result["condition_2"] is True


def test_conditions_dialog_empty_conditions(qtbot: QtBot) -> None:
    """Test dialog with empty conditions dictionary."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {}

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    assert dialog._list_widget.count() == 0
    assert dialog.get_conditions() == {}
