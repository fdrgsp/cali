"""Tests for evoked activity plotting functions."""

from __future__ import annotations

from cali.plot._multi_wells_plots._evoked_activity import (
    _aggregate_evoked_data_to_condition_stats,
)


def test_aggregate_evoked_data_to_condition_stats() -> None:
    """Test aggregation of evoked data with power/pulse flattened into names."""
    # Mock data structure: {condition: {well_id: {fov: {power_pulse: [amplitudes]}}}}
    data_by_condition = {
        "Control": {
            "w1": {
                "FOV_0": {
                    "5.0mW/cm²_50": [0.1, 0.2, 0.15],
                    "10.0mW/cm²_50": [0.2, 0.25, 0.22],
                }
            },
            "w2": {
                "FOV_1": {
                    "5.0mW/cm²_50": [0.12, 0.18],
                    "10.0mW/cm²_50": [0.21, 0.24],
                }
            },
        },
        "Treatment": {
            "w1": {
                "FOV_0": {
                    "5.0mW/cm²_50": [0.3, 0.35, 0.32],
                    "10.0mW/cm²_50": [0.4, 0.45, 0.42],
                }
            },
            "w2": {
                "FOV_1": {
                    "5.0mW/cm²_50": [0.28, 0.31],
                    "10.0mW/cm²_50": [0.38, 0.41],
                }
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
            "w1": {"FOV_0": {"5.0mW/cm²_50": [0.1, 0.2]}},
            "w2": {"FOV_1": {"5.0mW/cm²_50": [0.15]}},
        },
        "Treatment": {
            "w1": {"FOV_0": {"5.0mW/cm²_50": [0.3, 0.35]}},
            "w2": {"FOV_1": {"5.0mW/cm²_50": [0.28]}},
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
            "w1": {"FOV_0": {"10%_25": [0.1, 0.2]}},
            "w2": {"FOV_1": {"10%_25": [0.15, 0.18]}},
        },
        "Treatment": {
            "w1": {"FOV_2": {"10%_25": [0.3]}},
        },
    }

    plot_data = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    # Should extract "10%" as the power
    expected_conditions = {"Control (10%)", "Treatment (10%)"}
    assert set(plot_data["conditions"]) == expected_conditions
    assert len(plot_data["fov_values_list"]) == 2


def test_stim_split_amplitude_appends_evk_suffixes() -> None:
    """Aggregated stim/non-stim amplitude data gets EVK_STIM / EVK_NON_STIM suffixes.

    This exercises the combining logic inside
    plot_calcium_peaks_amplitude_stim_split_bar_plot without requiring a real DB.
    """
    from cali._constants import EVK_NON_STIM, EVK_STIM

    stim_data = {"ctrl": {"w1": {"FOV_0": {"25%_50": [0.1, 0.2]}}}}
    non_stim_data = {"ctrl": {"w1": {"FOV_1": {"25%_50": [0.05, 0.08]}}}}

    stim_plot = _aggregate_evoked_data_to_condition_stats(stim_data)
    non_stim_plot = _aggregate_evoked_data_to_condition_stats(non_stim_data)

    stim_conditions = [f"{c}_{EVK_STIM}" for c in stim_plot["conditions"]]
    non_stim_conditions = [f"{c}_{EVK_NON_STIM}" for c in non_stim_plot["conditions"]]

    # All stim conditions must end with EVK_STIM
    assert all(c.endswith(EVK_STIM) for c in stim_conditions)
    # All non-stim conditions must end with EVK_NON_STIM
    assert all(c.endswith(EVK_NON_STIM) for c in non_stim_conditions)
    # Power label is preserved in the condition name
    assert any("25%" in c for c in stim_conditions)
    assert any("25%" in c for c in non_stim_conditions)
    # No overlap between stim and non-stim condition names
    assert not set(stim_conditions) & set(non_stim_conditions)


def test_stim_split_amplitude_interleaved_order() -> None:
    """Bars are interleaved: stim/non-stim pairs grouped by condition+power.

    Expected x-axis order (for two conditions, two powers each):
      ctrl (10%)_evk_stim, ctrl (10%)_evk_non_stim,
      ctrl (25%)_evk_stim, ctrl (25%)_evk_non_stim,
      trt (10%)_evk_stim,  trt (10%)_evk_non_stim,
      trt (25%)_evk_stim,  trt (25%)_evk_non_stim
    NOT: all stim first, then all non-stim.
    """
    import re

    from cali._constants import EVK_NON_STIM, EVK_STIM

    stim_raw = {
        "ctrl": {"w1": {"FOV_0": {"10%_50": [0.1], "25%_50": [0.2]}}},
        "trt": {"w1": {"FOV_1": {"10%_50": [0.3], "25%_50": [0.4]}}},
    }
    non_stim_raw = {
        "ctrl": {"w1": {"FOV_0": {"10%_50": [0.05], "25%_50": [0.08]}}},
        "trt": {"w1": {"FOV_1": {"10%_50": [0.11], "25%_50": [0.15]}}},
    }

    stim_plot = _aggregate_evoked_data_to_condition_stats(stim_raw)
    non_stim_plot = _aggregate_evoked_data_to_condition_stats(non_stim_raw)

    # Reproduce the interleaving logic from the plot function
    stim_lookup = {
        c: (m, s, f)
        for c, m, s, f in zip(
            stim_plot["conditions"],
            stim_plot["means"],
            stim_plot["sems"],
            stim_plot["fov_values_list"],
        )
    }
    non_stim_lookup = {
        c: (m, s, f)
        for c, m, s, f in zip(
            non_stim_plot["conditions"],
            non_stim_plot["means"],
            non_stim_plot["sems"],
            non_stim_plot["fov_values_list"],
        )
    }

    seen: set[str] = set()
    all_base: list[str] = []
    for cond in list(stim_lookup) + list(non_stim_lookup):
        if cond not in seen:
            seen.add(cond)
            all_base.append(cond)

    def _sort_key(c: str) -> tuple[str, float]:
        base = c.rsplit(" (", 1)[0]
        tail = c.rsplit(" (", 1)[-1] if " (" in c else ""
        m_match = re.search(r"(\d+\.?\d*)", tail)
        return (base, float(m_match.group(1)) if m_match else 0.0)

    all_base.sort(key=_sort_key)

    combined: list[str] = []
    for base_cond in all_base:
        if base_cond in stim_lookup:
            combined.append(f"{base_cond}_{EVK_STIM}")
        if base_cond in non_stim_lookup:
            combined.append(f"{base_cond}_{EVK_NON_STIM}")

    expected = [
        f"ctrl (10%)_{EVK_STIM}",
        f"ctrl (10%)_{EVK_NON_STIM}",
        f"ctrl (25%)_{EVK_STIM}",
        f"ctrl (25%)_{EVK_NON_STIM}",
        f"trt (10%)_{EVK_STIM}",
        f"trt (10%)_{EVK_NON_STIM}",
        f"trt (25%)_{EVK_STIM}",
        f"trt (25%)_{EVK_NON_STIM}",
    ]
    assert combined == expected, f"Got: {combined}"


def test_stimulated_non_stimulated_amplitude_products_removed() -> None:
    """Standalone stim-only / non-stim-only amplitude plots must no longer exist.

    These were replaced by the combined Calcium Peaks Amplitude Bar Plot
    (Stim vs NonStim) which shows both sides in one chart with power labels.
    """
    from cali.plot._main_plot import ANALYSIS_PRODUCTS

    names = {p.name for p in ANALYSIS_PRODUCTS}
    assert "Stimulated Peaks Amplitude Bar Plot" not in names, (
        "Standalone stim-only amplitude plot should have been removed"
    )
    assert "Non-Stimulated Peaks Amplitude Bar Plot" not in names, (
        "Standalone non-stim-only amplitude plot should have been removed"
    )
