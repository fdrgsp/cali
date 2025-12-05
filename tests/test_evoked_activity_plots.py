"""Tests for evoked activity plotting functions."""

from __future__ import annotations

from cali.plot._multi_wells_plots._evoked_activity import (
    _aggregate_evoked_data_to_condition_stats,
)


def test_aggregate_evoked_data_to_condition_stats() -> None:
    """Test aggregation of evoked data with power/pulse flattened into names."""
    # Mock data structure: {condition: {fov: {power_pulse: [amplitudes]}}}
    data_by_condition = {
        "Control": {
            "FOV_0": {
                "5.0mW/cm²_50": [0.1, 0.2, 0.15],
                "10.0mW/cm²_50": [0.2, 0.25, 0.22],
            },
            "FOV_1": {
                "5.0mW/cm²_50": [0.12, 0.18],
                "10.0mW/cm²_50": [0.21, 0.24],
            },
        },
        "Treatment": {
            "FOV_0": {
                "5.0mW/cm²_50": [0.3, 0.35, 0.32],
                "10.0mW/cm²_50": [0.4, 0.45, 0.42],
            },
            "FOV_1": {
                "5.0mW/cm²_50": [0.28, 0.31],
                "10.0mW/cm²_50": [0.38, 0.41],
            },
        },
    }

    plot_data = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    # Should flatten power/pulse into condition names
    # Expected: Control (5.0mW/cm²), Treatment (5.0mW/cm²),
    #           Control (10.0mW/cm²), Treatment (10.0mW/cm²)
    expected_conditions = {
        "Control (5.0mW/cm²)",
        "Treatment (5.0mW/cm²)",
        "Control (10.0mW/cm²)",
        "Treatment (10.0mW/cm²)",
    }
    assert set(plot_data["conditions"]) == expected_conditions
    assert len(plot_data["means"]) == 4
    assert len(plot_data["sems"]) == 4
    assert len(plot_data["fov_values_list"]) == 4


def test_aggregate_evoked_data_single_power() -> None:
    """Test aggregation when all conditions use the same power/pulse."""
    data_by_condition = {
        "Control": {
            "FOV_0": {"5.0mW/cm²_50": [0.1, 0.2]},
            "FOV_1": {"5.0mW/cm²_50": [0.15]},
        },
        "Treatment": {
            "FOV_0": {"5.0mW/cm²_50": [0.3, 0.35]},
            "FOV_1": {"5.0mW/cm²_50": [0.28]},
        },
    }

    plot_data = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    # Should have conditions with power in names
    expected_conditions = {"Control (5.0mW/cm²)", "Treatment (5.0mW/cm²)"}
    assert set(plot_data["conditions"]) == expected_conditions
    assert len(plot_data["means"]) == 2
    assert len(plot_data["sems"]) == 2
    assert len(plot_data["fov_values_list"]) == 2


def test_aggregate_evoked_data_percentage_power() -> None:
    """Test aggregation with percentage-based power values."""
    data_by_condition = {
        "Control": {
            "FOV_0": {"10%_25": [0.1, 0.2]},
            "FOV_1": {"10%_25": [0.15, 0.18]},
        },
        "Treatment": {
            "FOV_2": {"10%_25": [0.3]},
        },
    }

    plot_data = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    # Should extract "10%" as the power
    expected_conditions = {"Control (10%)", "Treatment (10%)"}
    assert set(plot_data["conditions"]) == expected_conditions
    assert len(plot_data["fov_values_list"]) == 2
