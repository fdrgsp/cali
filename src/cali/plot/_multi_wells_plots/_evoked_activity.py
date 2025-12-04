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

        # Group by condition and FOV
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

            # Build condition label
            cond_label = _get_condition_label(well, fov.name)

            # Store amplitudes grouped by power_pulse
            for power_pulse, amplitude_list in amps.items():
                data.setdefault(cond_label, {}).setdefault(fov.name, {}).setdefault(
                    power_pulse, []
                ).extend(amplitude_list)

    return data


def _aggregate_evoked_data_to_condition_stats(
    data_by_condition: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, BarPlotData]:
    """Aggregate evoked amplitude data to condition-level statistics per power/pulse.

    Parameters
    ----------
    data_by_condition : dict[str, dict[str, dict[str, list[float]]]]
        Nested dict: {condition: {fov: {power_pulse: [amplitudes]}}}

    Returns
    -------
    dict[str, BarPlotData]
        Dict mapping power_pulse to aggregated plot data
    """
    # First, reorganize by power_pulse → condition → fov → values
    by_power_pulse: dict[str, dict[str, dict[str, list[float]]]] = {}

    for condition, fov_dict in data_by_condition.items():
        for fov, power_pulse_dict in fov_dict.items():
            for power_pulse, amplitudes in power_pulse_dict.items():
                by_power_pulse.setdefault(power_pulse, {}).setdefault(
                    condition, {}
                ).setdefault(fov, []).extend(amplitudes)

    # Now aggregate each power_pulse group
    result = {}
    for power_pulse, condition_data in by_power_pulse.items():
        result[power_pulse] = _aggregate_fov_data_to_condition_stats(condition_data)

    return result


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

    # Aggregate by power/pulse
    plot_data_by_power = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    if not plot_data_by_power:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # For now, plot the first power/pulse combination
    # TODO: Add UI to select which power/pulse to display
    power_pulse = next(iter(plot_data_by_power.keys()))
    plot_data = plot_data_by_power[power_pulse]

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="ΔF/F0",
        title_suffix=f" ({power_pulse})",
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

    # Aggregate by power/pulse
    plot_data_by_power = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    if not plot_data_by_power:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # For now, plot the first power/pulse combination
    # TODO: Add UI to select which power/pulse to display
    power_pulse = next(iter(plot_data_by_power.keys()))
    plot_data = plot_data_by_power[power_pulse]

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="ΔF/F0",
        title_suffix=f" ({power_pulse})",
        bar_label="Weighted Mean ± Pooled SEM",
    )
