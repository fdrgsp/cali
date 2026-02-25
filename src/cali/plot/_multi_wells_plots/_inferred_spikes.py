"""Inferred spikes and burst bar plots for multi-well analysis.

This module provides bar plot visualizations for inferred spike and burst metrics:
- Inferred spike frequency (thresholded spikes and rising edges)
- Burst count, average duration, average interval, and rate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, select

from cali.sqlmodel import FOV, Well

from ._util import (
    _aggregate_fov_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
    plot_parameter_bar_plot,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


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
        "avg_interval_sec": ..., "rate_per_min": ...}}}
    """
    from scipy.ndimage import gaussian_filter1d

    from cali.plot._single_wells_plots.burst._plot_burst_activity import (
        _detect_population_bursts,
        _get_burst_parameters,
        _get_population_spike_data,
    )

    # Get burst detection parameters
    burst_params = _get_burst_parameters(engine, fov_name="", rois=None, run_id=run_id)
    if burst_params is None:
        return {}

    burst_threshold, min_burst_duration_ms, smoothing_sigma_sec = burst_params

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

            # Compute frame rate from time axis
            num_frames = len(time_axis)
            if num_frames > 1:
                total_time_sec = float(time_axis[-1] - time_axis[0])
                frame_rate = (
                    (num_frames - 1) / total_time_sec if total_time_sec > 0 else 10.0
                )
            else:
                frame_rate = 10.0

            # Convert parameters to frame units
            min_burst_duration = max(
                1, int((min_burst_duration_ms / 1000.0) * frame_rate)
            )
            smoothing_sigma = smoothing_sigma_sec * frame_rate

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
                burst_metrics: dict[str, float] = {
                    "count": 0.0,
                    "avg_duration_sec": 0.0,
                    "avg_interval_sec": 0.0,
                    "rate_per_min": 0.0,
                }
            else:
                durations = []
                intervals = []

                for i, (start, end) in enumerate(bursts):
                    duration_sec = (end - start) * (time_axis[1] - time_axis[0])
                    durations.append(duration_sec)

                    if i < len(bursts) - 1:
                        next_start = bursts[i + 1][0]
                        interval_sec = (next_start - end) * (
                            time_axis[1] - time_axis[0]
                        )
                        intervals.append(interval_sec)

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


def plot_burst_count_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst count across conditions."""
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    count_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        count_data[cond] = {
            fov: [metrics["count"]] for fov, metrics in fov_dict.items()
        }

    plot_data = _aggregate_fov_data_to_condition_stats(count_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

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
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    duration_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        duration_data[cond] = {
            fov: [metrics["avg_duration_sec"]] for fov, metrics in fov_dict.items()
        }

    plot_data = _aggregate_fov_data_to_condition_stats(duration_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

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
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    interval_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        interval_data[cond] = {
            fov: [metrics["avg_interval_sec"]] for fov, metrics in fov_dict.items()
        }

    plot_data = _aggregate_fov_data_to_condition_stats(interval_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

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
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    rate_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        rate_data[cond] = {
            fov: [metrics["rate_per_min"]] for fov, metrics in fov_dict.items()
        }

    plot_data = _aggregate_fov_data_to_condition_stats(rate_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="bursts/min",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_inferred_spikes_frequency_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes frequency (thresholded) across conditions.

    Uses ``DataAnalysis.inferred_spikes_frequency`` (per active ROI).
    Aggregation: per-ROI scalar → FOV weighted mean → condition pooled SEM.
    """
    plot_parameter_bar_plot(
        widget,
        text,
        engine,
        run_id,
        parameter="inferred_spikes_frequency",
        units="Hz",
        title_suffix=" (thresholded spikes)",
    )


def plot_inferred_spikes_rising_edge_frequency_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes frequency (rising edges) across conditions.

    Uses ``DataAnalysis.inferred_spikes_rising_edge_frequency`` (per active ROI).
    Aggregation: per-ROI scalar → FOV weighted mean → condition pooled SEM.
    """
    plot_parameter_bar_plot(
        widget,
        text,
        engine,
        run_id,
        parameter="inferred_spikes_rising_edge_frequency",
        units="Hz",
        title_suffix=" (rising edges)",
    )
