"""Tests for condition color defaults and customization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtWidgets import QWidget

from cali.gui._pygraph_plot_widgets import _ConditionsDialog
from cali.plot._multi_wells_plots._util import (
    _get_default_color,
    _get_default_conditions,
)

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.mark.parametrize(
    ("condition_name", "expected_color"),
    [
        # evk_stim conditions should get green
        ("c1_g1_evk_stim", "green"),
        ("control_evk_stim", "green"),
        ("treatment_A_evk_stim", "green"),
        # evk_non_stim conditions should get magenta
        ("c1_g1_evk_non_stim", "magenta"),
        ("control_evk_non_stim", "magenta"),
        ("treatment_A_evk_non_stim", "magenta"),
        # non-evoked conditions should get gray
        ("control", "gray"),
        ("treatment_A", "gray"),
        ("knockout", "gray"),
        ("c1_g1", "gray"),
    ],
)
def test_default_color(condition_name: str, expected_color: str) -> None:
    """Test default color assignment for various condition names."""
    assert _get_default_color(condition_name) == expected_color


def test_get_default_conditions() -> None:
    """Test that _get_default_conditions creates correct structure."""
    conditions = ["control", "treatment_evk_stim", "treatment_evk_non_stim"]
    result = _get_default_conditions(conditions)

    # Check structure
    assert len(result) == 3
    assert all("visible" in v and "color" in v for v in result.values())

    # Check all are visible
    assert all(v["visible"] for v in result.values())

    # Check colors
    assert result["control"]["color"] == "gray"
    assert result["treatment_evk_stim"]["color"] == "green"
    assert result["treatment_evk_non_stim"]["color"] == "magenta"


def test_get_default_conditions_order() -> None:
    """Test that _get_default_conditions preserves input order."""
    conditions = ["D", "C", "B", "A"]
    result = _get_default_conditions(conditions)

    assert list(result.keys()) == conditions


def test_dialog_stores_colors(qtbot: QtBot) -> None:
    """Test that dialog stores and retrieves colors correctly."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": {"visible": True, "color": "gray"},
        "treatment_evk_stim": {"visible": True, "color": "green"},
        "treatment_evk_non_stim": {"visible": False, "color": "magenta"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Get conditions back
    result = dialog.get_conditions()

    # Check colors are preserved
    assert result["control"]["color"] == "gray"
    assert result["treatment_evk_stim"]["color"] == "green"
    assert result["treatment_evk_non_stim"]["color"] == "magenta"


def test_dialog_preserves_colors_after_reorder(qtbot: QtBot) -> None:
    """Test that colors are preserved when reordering items."""
    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": {"visible": True, "color": "gray"},
        "treatment_evk_stim": {"visible": True, "color": "green"},
        "treatment_evk_non_stim": {"visible": True, "color": "magenta"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Reorder: move first item to last
    item = dialog._list_widget.item(0)
    widget = dialog._list_widget.itemWidget(item)
    dialog._list_widget.takeItem(0)
    dialog._list_widget.addItem(item)
    dialog._list_widget.setItemWidget(item, widget)

    result = dialog.get_conditions()

    # Order should be changed
    assert list(result.keys()) == [
        "treatment_evk_stim",
        "treatment_evk_non_stim",
        "control",
    ]

    # Colors should be preserved
    assert result["control"]["color"] == "gray"
    assert result["treatment_evk_stim"]["color"] == "green"
    assert result["treatment_evk_non_stim"]["color"] == "magenta"


def test_mixed_evoked_non_evoked_defaults() -> None:
    """Test default colors for mixed evoked and non-evoked conditions."""
    conditions = [
        "control",
        "control_evk_stim",
        "control_evk_non_stim",
        "treatment",
        "treatment_evk_stim",
        "treatment_evk_non_stim",
    ]

    result = _get_default_conditions(conditions)

    # Non-evoked conditions should be gray
    assert result["control"]["color"] == "gray"
    assert result["treatment"]["color"] == "gray"

    # Stimulated conditions should be green
    assert result["control_evk_stim"]["color"] == "green"
    assert result["treatment_evk_stim"]["color"] == "green"

    # Non-stimulated conditions should be magenta
    assert result["control_evk_non_stim"]["color"] == "magenta"
    assert result["treatment_evk_non_stim"]["color"] == "magenta"


def test_dialog_color_change(qtbot: QtBot) -> None:
    """Test that changing color in dialog is reflected in get_conditions."""
    from cali.gui._pygraph_plot_widgets import _ConditionItemWidget

    parent = QWidget()
    qtbot.addWidget(parent)

    conditions = {
        "control": {"visible": True, "color": "gray"},
        "treatment": {"visible": True, "color": "green"},
    }

    dialog = _ConditionsDialog(conditions, parent)
    qtbot.addWidget(dialog)

    # Change color of first item
    item0 = dialog._list_widget.item(0)
    widget0 = dialog._list_widget.itemWidget(item0)
    assert isinstance(widget0, _ConditionItemWidget)
    widget0._color_combo.setCurrentText("red")

    # Get conditions
    result = dialog.get_conditions()

    # Color should be changed
    assert result["control"]["color"] == "red"
    assert result["treatment"]["color"] == "green"
