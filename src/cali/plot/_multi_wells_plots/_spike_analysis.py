"""Spike analysis bar plots for multi-well analysis.

This module provides bar plot visualizations for spike and burst metrics:
- Spike synchrony
- Burst count, duration, interval, and rate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, col, select

from cali.sqlmodel import FOV, ROI, DataAnalysis, Traces, Well

from ._util import (
    _aggregate_fov_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


def _query_spike_synchrony_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, float]]:
    """Query spike synchrony per FOV, grouped by condition.

    Calculates synchrony on-the-fly from inferred spikes.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {condition: {fov_name: synchrony_value}}
    """
    from cali.plot._util import _get_spike_synchrony, _get_spike_synchrony_matrix

    with Session(engine) as session:
        # Query all ROIs with traces and analysis grouped by FOV
        stmt = (
            select(ROI, FOV, Well, Traces, DataAnalysis)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Traces, ROI.id == Traces.roi_id)
            .join(DataAnalysis, ROI.id == DataAnalysis.roi_id)
        )

        if run_id is not None:
            stmt = stmt.where(col(Traces.analysis_result_id) == run_id).where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        # Only active ROIs
        stmt = stmt.where(col(ROI.active) == True)  # noqa: E712
        stmt = stmt.order_by(col(FOV.id), col(ROI.label_value))

        results = session.exec(stmt).all()

        # Group by condition and FOV
        data: dict[str, dict[str, float]] = {}

        # Organize by FOV first
        fov_data: dict[tuple[str, str], list[tuple[ROI, Traces, DataAnalysis]]] = {}
        for roi, fov, well, traces, analysis in results:
            cond_label = _get_condition_label(well, fov.name)
            key = (cond_label, fov.name)
            fov_data.setdefault(key, []).append((roi, traces, analysis))

        # Calculate synchrony for each FOV
        for (cond_label, fov_name), roi_list in fov_data.items():
            if len(roi_list) < 2:
                # Need at least 2 ROIs for synchrony
                continue

            # Build spike data dict (thresholded spikes)
            spike_data: dict[str, list[float]] = {}
            for roi, traces, analysis in roi_list:
                if (
                    not traces.inferred_spikes
                    or analysis.inferred_spikes_threshold is None
                ):
                    continue

                # Threshold the spikes
                inferred_spikes = np.array(traces.inferred_spikes)
                threshold = analysis.inferred_spikes_threshold
                spikes_thresholded = np.where(
                    inferred_spikes > threshold, inferred_spikes, 0.0
                ).tolist()

                roi_key = f"ROI_{roi.label_value}"
                spike_data[roi_key] = spikes_thresholded

            if len(spike_data) < 2:
                continue

            # Calculate synchrony matrix
            sync_matrix = _get_spike_synchrony_matrix(spike_data)

            # Calculate global synchrony (median)
            if sync_matrix is not None:
                global_sync = _get_spike_synchrony(sync_matrix)

                if global_sync is not None:
                    data.setdefault(cond_label, {})[fov_name] = global_sync

    return data


def _query_burst_metrics_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Query burst metrics per FOV, grouped by condition.

    Calculates burst count, avg duration, and avg interval on-the-fly
    from population spike activity.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Nested dict: {condition: {fov_name: {"count": ..., "avg_duration_sec": ...,
        "avg_interval_sec": ...}}}
    """
    from scipy.ndimage import gaussian_filter1d

    from cali.plot._single_wells_plots.burst._plot_inferred_spike_burst_activity import (  # noqa: E501
        _detect_population_bursts,
        _get_burst_parameters,
        _get_population_spike_data,
    )

    # Get burst detection parameters
    burst_params = _get_burst_parameters(engine, fov_name="", rois=None, run_id=run_id)
    if burst_params is None:
        return {}

    burst_threshold, min_burst_duration, smoothing_sigma = burst_params

    with Session(engine) as session:
        # Get all FOV names grouped by condition
        stmt = (
            select(FOV, Well)
            .select_from(FOV)
            .join(Well, FOV.well_id == Well.id)
            .distinct()
        )
        fov_well_results = session.exec(stmt).all()

        data: dict[str, dict[str, dict[str, float]]] = {}

        for fov, well in fov_well_results:
            cond_label = _get_condition_label(well)

            # Get population spike data for this FOV
            spike_trains, _, time_axis = _get_population_spike_data(
                engine, fov.name, rois=None, run_id=run_id
            )

            if spike_trains is None or len(spike_trains) < 2:
                continue

            # Calculate population activity
            population_activity = np.mean(spike_trains, axis=0)

            # Smooth population activity for burst detection
            smoothed_activity = gaussian_filter1d(
                population_activity, sigma=smoothing_sigma
            )

            # Detect bursts (threshold is percentage, convert to fraction)
            bursts = _detect_population_bursts(
                smoothed_activity, burst_threshold / 100, min_burst_duration
            )

            # Calculate burst statistics
            burst_count = len(bursts)

            # Calculate burst rate (bursts per minute)
            total_time_min = (time_axis[-1] - time_axis[0]) / 60.0
            burst_rate = burst_count / total_time_min if total_time_min > 0 else 0.0

            if burst_count == 0:
                # No bursts detected
                burst_metrics = {
                    "count": 0.0,
                    "avg_duration_sec": 0.0,
                    "avg_interval_sec": 0.0,
                    "rate_per_min": 0.0,
                }
            else:
                # Calculate durations and intervals
                durations = []
                intervals = []

                for i, (start, end) in enumerate(bursts):
                    # Convert indices to time
                    duration_sec = (end - start) * (time_axis[1] - time_axis[0])
                    durations.append(duration_sec)

                    # Calculate interval to next burst
                    if i < len(bursts) - 1:
                        next_start = bursts[i + 1][0]
                        interval_sec = (next_start - end) * (
                            time_axis[1] - time_axis[0]
                        )
                        intervals.append(interval_sec)

                # Calculate statistics
                avg_duration = float(np.mean(durations)) if durations else 0.0
                avg_interval = float(np.mean(intervals)) if intervals else 0.0

                burst_metrics = {
                    "count": float(burst_count),
                    "avg_duration_sec": avg_duration,
                    "avg_interval_sec": avg_interval,
                    "rate_per_min": float(burst_rate),
                }

            data.setdefault(cond_label, {})[fov.name] = burst_metrics

    return data


def plot_spike_synchrony_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes global synchrony across conditions."""
    # Query synchrony data (one value per FOV)
    data_by_condition = _query_spike_synchrony_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Convert to format expected by aggregation
    # Since we have single values per FOV, wrap in lists
    data_as_lists: dict[str, dict[str, list[float]]] = {}
    for condition, fov_dict in data_by_condition.items():
        for fov_name, sync_value in fov_dict.items():
            data_as_lists.setdefault(condition, {})[fov_name] = [sync_value]

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(data_as_lists)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="Index",
        title_suffix=" (Median - Thresholded Data)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_count_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst count across conditions."""
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract burst count from metrics dict
    count_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        count_data[cond] = {
            fov: [metrics["count"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(count_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="Count",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_avg_duration_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst average duration across conditions."""
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract avg duration from metrics dict
    duration_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        duration_data[cond] = {
            fov: [metrics["avg_duration_sec"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(duration_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="s",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_avg_interval_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst average interval across conditions."""
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract avg interval from metrics dict
    interval_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        interval_data[cond] = {
            fov: [metrics["avg_interval_sec"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(interval_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="s",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_rate_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst rate across conditions."""
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract burst rate from metrics dict
    rate_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        rate_data[cond] = {
            fov: [metrics["rate_per_min"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(rate_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="bursts/min",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )
