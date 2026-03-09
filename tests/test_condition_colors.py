"""Tests for condition color defaults and customization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtWidgets import QWidget

from cali.gui._pygraph_plot_widgets import _ConditionsDialog
from cali.plot._multi_wells_plots._util import (
    _get_default_color,
    _get_default_conditions,
    _make_n_colors,
)
from cali.plot._single_wells_plots.cluster._plot_cluster_analysis import (
    _make_n_cluster_colors,
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

    # EVK conditions keep their fixed colors
    assert result["treatment_evk_stim"]["color"] == "green"
    assert result["treatment_evk_non_stim"]["color"] == "magenta"

    # Non-EVK condition gets gray (bar-plot default: neutral gray)
    assert result["control"]["color"] == "gray"


def test_get_default_conditions_order() -> None:
    """Test that _get_default_conditions preserves input order."""
    conditions = ["D", "C", "B", "A"]
    result = _get_default_conditions(conditions)

    assert list(result.keys()) == conditions


def test_get_default_conditions_distinct_palette_colors() -> None:
    """Test that multiple non-EVK conditions get distinct colors with
    multicolor=True.
    """
    conditions = ["g1_t1", "g1_t2", "g2_t1", "g2_t2"]
    result = _get_default_conditions(conditions, multicolor=True)

    colors = [result[c]["color"] for c in conditions]
    # All non-EVK conditions must be assigned distinct colors in multicolor mode
    assert len(set(colors)) == len(colors), (
        f"Expected {len(conditions)} distinct colors, got: {colors}"
    )


def test_get_default_conditions_all_gray_by_default() -> None:
    """Non-EVK conditions are all gray by default (bar-plot mode)."""
    conditions = ["g1_t1", "g1_t2", "g2_t1", "g2_t2"]
    result = _get_default_conditions(conditions)

    for cond in conditions:
        assert result[cond]["color"] == "gray", (
            f"Expected gray for {cond}, got {result[cond]['color']}"
        )


def test_get_default_conditions_override_color_applies_to_all() -> None:
    """When override_color is given, all conditions receive that color."""
    conditions = ["ctrl", "drug", "vehicle"]
    result = _get_default_conditions(conditions, override_color="green")

    for cond in conditions:
        assert result[cond]["color"] == "green", (
            f"Expected green for {cond}, got {result[cond]['color']}"
        )


def test_get_default_conditions_override_color_supersedes_evk() -> None:
    """override_color takes precedence even over EVK_STIM / EVK_NON_STIM conditions."""
    conditions = ["ctrl_evk_stim", "ctrl_evk_non_stim", "control"]
    result = _get_default_conditions(conditions, override_color="magenta")

    for cond in conditions:
        assert result[cond]["color"] == "magenta", (
            f"Expected magenta for {cond}, got {result[cond]['color']}"
        )


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

    # Bar-plot mode (default): non-EVK conditions are gray
    result = _get_default_conditions(conditions)
    assert result["control"]["color"] == "gray"
    assert result["treatment"]["color"] == "gray"

    # Multicolor mode (PCA): each non-EVK condition gets a distinct color
    result_multi = _get_default_conditions(conditions, multicolor=True)
    assert result_multi["control"]["color"] != "gray"
    assert result_multi["treatment"]["color"] != "gray"
    assert result_multi["control"]["color"] != result_multi["treatment"]["color"]

    # Stimulated conditions are always green
    assert result["control_evk_stim"]["color"] == "green"
    assert result["treatment_evk_stim"]["color"] == "green"

    # Non-stimulated conditions are always magenta
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


@pytest.mark.parametrize("n", [1, 5, 10, 11, 15, 25])
def test_make_n_colors_returns_n_unique_colors(n: int) -> None:
    """Test that _make_n_colors returns exactly n unique hex color strings."""
    colors = _make_n_colors(n)
    assert len(colors) == n, f"Expected {n} colors, got {len(colors)}"
    assert len(set(colors)) == n, (
        f"Expected {n} unique colors for n={n}, got {len(set(colors))}: {colors}"
    )


def test_make_n_colors_empty() -> None:
    """Test that _make_n_colors returns empty list for n=0."""
    assert _make_n_colors(0) == []


def test_get_default_conditions_more_than_palette_size() -> None:
    """Test that _get_default_conditions assigns unique colors for > 10
    conditions (multicolor=True).
    """
    # Create 13 non-EVK conditions (more than the 10-color palette)
    conditions = [f"group_{i}" for i in range(13)]
    result = _get_default_conditions(conditions, multicolor=True)

    colors = [result[c]["color"] for c in conditions]
    assert len(set(colors)) == len(colors), (
        f"All {len(conditions)} conditions must have unique colors, got: {colors}"
    )


@pytest.mark.parametrize("n", [1, 5, 10, 11, 15, 25])
def test_make_n_cluster_colors_returns_n_unique_colors(n: int) -> None:
    """Test that _make_n_cluster_colors returns exactly n unique RGBA tuples."""
    colors = _make_n_cluster_colors(n)
    assert len(colors) == n, f"Expected {n} cluster colors, got {len(colors)}"
    assert len(set(colors)) == n, (
        f"Expected {n} unique cluster colors for n={n},"
        f" got {len(set(colors))}: {colors}"
    )


def test_make_n_cluster_colors_empty() -> None:
    """Test that _make_n_cluster_colors returns empty list for n=0."""
    assert _make_n_cluster_colors(0) == []
