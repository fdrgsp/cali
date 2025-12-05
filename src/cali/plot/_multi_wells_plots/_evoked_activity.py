"""Evoked activity bar plots for multi-well analysis.

This module provides bar plot visualizations for evoked experiments:
- Stimulated peaks amplitude
- Non-stimulated peaks amplitude
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, col, select

from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    Traces,
    Well,
)

from ._util import (
    BarPlotData,
    _aggregate_fov_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


def _query_evoked_amplitudes_by_condition(
    engine: Engine,
    stimulated: bool = True,
    run_id: int | None = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Query evoked amplitudes grouped by condition → FOV → power_pulse.

    Calculates stimulated/non-stimulated amplitudes on-the-fly from traces data.

    Parameters
    ----------
    engine : Engine
        Database engine
    stimulated : bool
        If True, get stimulated peaks; if False, get non-stimulated peaks
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, dict[str, list[float]]]]
        Nested dict: {condition: {fov: {power_pulse: [amplitudes]}}}
    """
    from cali.plot._util import separate_stimulated_vs_non_stimulated_peaks

    with Session(engine) as session:
        # Build query for all ROIs with their traces and analysis
        stmt = (
            select(ROI, FOV, Well, Traces, DataAnalysis, AnalysisSettings)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Traces, ROI.id == Traces.roi_id)
            .join(DataAnalysis, ROI.id == DataAnalysis.roi_id)
            .join(CaliResult, Traces.analysis_result_id == CaliResult.id)
            .join(
                AnalysisSettings, CaliResult.analysis_settings_id == AnalysisSettings.id
            )
        )

        if run_id is not None:
            stmt = stmt.where(col(Traces.analysis_result_id) == run_id).where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        # Only get active ROIs
        stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

        results = session.exec(stmt).all()

        # Group by condition and FOV and power_pulse
        data: dict[str, dict[str, dict[str, list[float]]]] = {}

        for roi, fov, well, traces, analysis, settings in results:
            # Check if this is an evoked experiment
            if not settings.led_pulse_on_frames or not settings.led_pulse_powers:
                continue

            # Build stimulations_frames_and_powers dict
            # Convert frames to int first to handle float values from database
            stimulations_frames_and_powers = {
                str(int(frame)): power
                for frame, power in zip(
                    settings.led_pulse_on_frames, settings.led_pulse_powers
                )
            }

            # Get stimulated/non-stimulated amplitudes
            amps_stim, amps_non_stim = separate_stimulated_vs_non_stimulated_peaks(
                dec_dff=np.array(traces.dec_dff),
                peaks_dec_dff=(
                    np.array(analysis.peaks_dec_dff)
                    if analysis.peaks_dec_dff
                    else np.array([])
                ),
                pulse_on_frames_and_powers=stimulations_frames_and_powers,
                is_roi_stimulated=roi.stimulated,
                led_pulse_duration=settings.led_pulse_duration or "unknown",
                led_power_equation=None,
            )

            # Select which dict to use
            amps = amps_stim if stimulated else amps_non_stim
            if not amps:
                continue

            # Build condition label (without power/pulse)
            cond_label = _get_condition_label(well, fov.name)

            # Store amplitudes grouped by power_pulse
            for power_pulse, amplitude_list in amps.items():
                data.setdefault(cond_label, {}).setdefault(fov.name, {}).setdefault(
                    power_pulse, []
                ).extend(amplitude_list)

    return data


def _aggregate_evoked_data_to_condition_stats(
    data_by_condition: dict[str, dict[str, dict[str, list[float]]]],
) -> BarPlotData:
    """Aggregate evoked amplitude data across all power/pulse combinations.

    Flattens power/pulse combinations into condition names for unified plotting.
    For example, "Control" with powers [2.0%, 4.0%] becomes:
    ["Control (2.0%)", "Control (4.0%)"]

    Results are ordered first by condition name, then by LED power (ascending).

    Parameters
    ----------
    data_by_condition : dict[str, dict[str, dict[str, list[float]]]]
        Nested dict: {condition: {fov: {power_pulse: [amplitudes]}}}

    Returns
    -------
    BarPlotData
        Aggregated plot data with power/pulse in condition names
    """
    # Reorganize to flatten structure: condition_power → fov → values
    # Also track the numeric power value for sorting
    flattened: dict[str, dict[str, list[float]]] = {}
    power_values: dict[str, float] = {}  # Maps condition_with_power to numeric power

    for condition, fov_dict in data_by_condition.items():
        for fov, power_pulse_dict in fov_dict.items():
            for power_pulse, amplitudes in power_pulse_dict.items():
                # Extract power value from power_pulse string
                # Format is "X.X%" or "X.XXXmW/cm²", followed by "_duration"
                power_str = power_pulse.split("_")[0]  # Get just the power part

                # Extract numeric value for sorting
                # Handle both "5.0%" and "5.000mW/cm²" formats
                import re

                numeric_match = re.search(r"(\d+\.?\d*)", power_str)
                numeric_power = float(numeric_match.group(1)) if numeric_match else 0.0

                # Create new condition name with power
                condition_with_power = f"{condition} ({power_str})"

                flattened.setdefault(condition_with_power, {}).setdefault(
                    fov, []
                ).extend(amplitudes)

                # Store the numeric power for sorting
                power_values[condition_with_power] = numeric_power

    # Sort by condition name first, then by power value
    # Extract base condition name (everything before the last opening parenthesis)
    def sort_key(cond_with_power: str) -> tuple[str, float]:
        base_condition = cond_with_power.rsplit(" (", 1)[0]
        power = power_values.get(cond_with_power, 0.0)
        return (base_condition, power)

    sorted_conditions = sorted(flattened.keys(), key=sort_key)

    # Rebuild flattened dict in sorted order
    sorted_flattened = {cond: flattened[cond] for cond in sorted_conditions}

    # Aggregate all flattened conditions
    return _aggregate_fov_data_to_condition_stats(sorted_flattened)


def plot_stimulated_peaks_amplitude_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot stimulated calcium peaks amplitude across conditions.

    For evoked experiments. Creates separate plots for each LED power/pulse
    combination.
    """
    # Query stimulated amplitudes
    data_by_condition = _query_evoked_amplitudes_by_condition(
        engine, stimulated=True, run_id=run_id
    )

    if not data_by_condition:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # Aggregate data (flattens power/pulse into condition names)
    plot_data = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    if not plot_data["conditions"]:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="ΔF/F0",
        title_suffix="",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_non_stimulated_peaks_amplitude_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot non-stimulated calcium peaks amplitude across conditions.

    For evoked experiments. Creates separate plots for each LED power/pulse
    combination.
    """
    # Query non-stimulated amplitudes
    data_by_condition = _query_evoked_amplitudes_by_condition(
        engine, stimulated=False, run_id=run_id
    )

    if not data_by_condition:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # Aggregate data (flattens power/pulse into condition names)
    plot_data = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    if not plot_data["conditions"]:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="ΔF/F0",
        title_suffix="",
        bar_label="Weighted Mean ± Pooled SEM",
    )
