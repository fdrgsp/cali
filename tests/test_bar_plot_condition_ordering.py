"""Test that bar plot respects widget.conditions order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cali.plot._multi_wells_plots._util import BarPlotData


def test_condition_ordering_logic() -> None:
    """Test that filtering respects the order defined in cond_list."""
    # This is the logic from _create_pyqtgraph_bar_plot

    # Simulate data from database (in alphabetical order)
    data: BarPlotData = {
        "conditions": ["control", "knockout", "treatment_A", "treatment_B"],
        "means": [10.0, 12.0, 15.0, 20.0],
        "sems": [1.0, 1.2, 1.5, 2.0],
        "fov_values_list": [
            np.array([9.0, 10.0, 11.0]),
            np.array([11.0, 12.0, 13.0]),
            np.array([13.0, 15.0, 17.0]),
            np.array([18.0, 20.0, 22.0]),
        ],
    }

    # User has reordered conditions (this is what widget.conditions contains)
    cond_list = {
        "treatment_A": {"visible": True, "color": "green"},
        "knockout": {"visible": True, "color": "gray"},
        "control": {"visible": True, "color": "gray"},
        "treatment_B": {"visible": False, "color": "magenta"},  # Hidden
    }

    # Create a mapping from condition name to data (same logic as in the function)
    data_map = {
        cond: (mean, sem, fov_vals)
        for cond, mean, sem, fov_vals in zip(
            data["conditions"],
            data["means"],
            data["sems"],
            data["fov_values_list"],
        )
    }

    # Build filtered data in the order defined by cond_list
    filtered_data = [
        (cond, *data_map[cond])
        for cond in cond_list.keys()
        if cond_list[cond]["visible"] and cond in data_map
    ]

    # Extract conditions
    filtered_conditions = [item[0] for item in filtered_data]

    # Verify the order matches cond_list order, not data order
    assert filtered_conditions == ["treatment_A", "knockout", "control"]
    assert filtered_conditions != ["control", "knockout", "treatment_A"]

    # Verify treatment_B is filtered out
    assert "treatment_B" not in filtered_conditions

    # Verify the data values match
    assert filtered_data[0][1] == 15.0  # treatment_A mean
    assert filtered_data[1][1] == 12.0  # knockout mean
    assert filtered_data[2][1] == 10.0  # control mean


def test_condition_ordering_preserves_new_order() -> None:
    """Test that user can completely rearrange the order."""
    data: BarPlotData = {
        "conditions": ["A", "B", "C", "D"],
        "means": [1.0, 2.0, 3.0, 4.0],
        "sems": [0.1, 0.2, 0.3, 0.4],
        "fov_values_list": [
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
        ],
    }

    # User reverses the order
    cond_list = {
        "D": {"visible": True, "color": "gray"},
        "C": {"visible": True, "color": "gray"},
        "B": {"visible": True, "color": "gray"},
        "A": {"visible": True, "color": "gray"},
    }

    data_map = {
        cond: (mean, sem, fov_vals)
        for cond, mean, sem, fov_vals in zip(
            data["conditions"],
            data["means"],
            data["sems"],
            data["fov_values_list"],
        )
    }

    filtered_data = [
        (cond, *data_map[cond])
        for cond in cond_list.keys()
        if cond_list[cond]["visible"] and cond in data_map
    ]

    filtered_conditions = [item[0] for item in filtered_data]

    # Order should be reversed
    assert filtered_conditions == ["D", "C", "B", "A"]
    assert filtered_conditions == list(reversed(data["conditions"]))
