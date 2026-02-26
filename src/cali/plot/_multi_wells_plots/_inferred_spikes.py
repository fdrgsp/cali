"""Inferred spikes and burst bar plots for multi-well analysis.

This module provides bar plot visualizations for inferred spike and burst metrics:
- Inferred spike frequency (thresholded spikes and rising edges)
- Burst count, average duration, average interval, and rate
- Spike synchrony and correlation across conditions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
    """Query pre-computed spike burst metrics from FOVAnalysis, grouped by condition.

    Uses stored `FOVAnalysis.spike_burst_count`, `spike_burst_avg_duration`,
    and `spike_burst_avg_interval` — the same values highlighted in the
    single-well burst view.  FOVs with no detected bursts (count is None or 0)
    are excluded, matching the behaviour of the calcium burst bar plots.

    Parameters
    ----------
    engine : Engine
        Database engine.
    run_id : int | None
        Filter by specific analysis run.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Nested dict: {condition: {fov_name: {"count": ..., "avg_duration_sec": ...,
        "avg_interval_sec": ..., "rate_per_min": ...}}}
    """
    from sqlalchemy.exc import OperationalError
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, FOVAnalysis, Well
    from cali.sqlmodel._model import AnalysisSettings, CaliResult

    try:
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis, FOV, Well, AnalysisSettings)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .join(Well, FOV.well_id == Well.id)
                .join(CaliResult, FOVAnalysis.analysis_result_id == CaliResult.id)
                .join(
                    AnalysisSettings,
                    CaliResult.analysis_settings_id == AnalysisSettings.id,
                )
            )
            if run_id is not None:
                stmt = stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)
            results = session.exec(stmt).all()

            data: dict[str, dict[str, dict[str, float]]] = {}

            for fa, fov, well, settings in results:
                if not fa.spike_burst_count:  # skip None and 0
                    continue

                cond_label = _get_condition_label(well)

                # Compute burst rate from stored population activity length
                rate_per_min = 0.0
                if fa.spike_population_activity and settings.frame_rate:
                    n_frames = len(fa.spike_population_activity)
                    duration_min = n_frames / settings.frame_rate / 60.0
                    if duration_min > 0:
                        rate_per_min = fa.spike_burst_count / duration_min

                data.setdefault(cond_label, {})[fov.name] = {
                    "count": float(fa.spike_burst_count),
                    "avg_duration_sec": (
                        float(fa.spike_burst_avg_duration)
                        if fa.spike_burst_avg_duration is not None
                        else 0.0
                    ),
                    "avg_interval_sec": (
                        float(fa.spike_burst_avg_interval)
                        if fa.spike_burst_avg_interval is not None
                        else 0.0
                    ),
                    "rate_per_min": rate_per_min,
                }

        return data
    except OperationalError:
        return {}


def _plot_burst_metric(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None,
    metric_key: str,
    units: str,
) -> None:
    """Plot a single burst metric across conditions.

    NOTE:
    metric_key: Key in the burst metrics dict (e.g. `"count"`, `"avg_duration_sec"`).
    units: Y-axis units label.
    """
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    metric_data: dict[str, dict[str, list[float]]] = {
        cond: {fov: [m[metric_key]] for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
    }

    plot_data = _aggregate_fov_data_to_condition_stats(metric_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units=units,
        title_suffix=" (Inferred Spikes)",
        bar_label="Mean ± SEM (per FOV)",
    )


def plot_burst_count_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst count across conditions."""
    _plot_burst_metric(widget, text, engine, run_id, "count", "Count")


def plot_burst_avg_duration_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst average duration across conditions."""
    _plot_burst_metric(widget, text, engine, run_id, "avg_duration_sec", "s")


def plot_burst_avg_interval_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst average interval across conditions."""
    _plot_burst_metric(widget, text, engine, run_id, "avg_interval_sec", "s")


def plot_burst_rate_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst rate across conditions."""
    _plot_burst_metric(widget, text, engine, run_id, "rate_per_min", "bursts/min")


def plot_inferred_spikes_frequency_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes frequency (thresholded) across conditions.

    Uses `DataAnalysis.inferred_spikes_frequency` (per active ROI).
    Aggregation: per-ROI scalar → FOV mean → condition mean ± SEM (per FOV).
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

    Uses `DataAnalysis.inferred_spikes_rising_edge_frequency` (per active ROI).
    Aggregation: per-ROI scalar → FOV mean → condition mean ± SEM (per FOV).
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


def plot_inferred_spikes_frequency_stim_split_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes frequency split by stim/non-stim within each condition.

    Evoked-only: condition labels are suffixed with '(Stim)' or '(NonStim)'.
    """
    plot_parameter_bar_plot(
        widget,
        text,
        engine,
        run_id,
        parameter="inferred_spikes_frequency",
        units="Hz",
        title_suffix=" (thresholded spikes)",
        include_stim_status=True,
    )


def plot_inferred_spikes_rising_edge_frequency_stim_split_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes rising edge frequency split by stim/non-stim.

    Evoked-only: condition labels are suffixed with '(Stim)' or '(NonStim)'.
    """
    plot_parameter_bar_plot(
        widget,
        text,
        engine,
        run_id,
        parameter="inferred_spikes_rising_edge_frequency",
        units="Hz",
        title_suffix=" (rising edges)",
        include_stim_status=True,
    )


# ---------------------------------------------------------------------------
# Spike synchrony and correlation bar plots
# ---------------------------------------------------------------------------


def _query_spike_synchrony_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, float]]:
    """Query spike synchrony per FOV, grouped by condition.

    Uses pre-computed global_spike_jitter_synchrony from FOVAnalysis.

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
    from sqlalchemy.exc import OperationalError
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, FOVAnalysis, Well

    try:
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis, FOV, Well)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .join(Well, FOV.well_id == Well.id)
            )

            if run_id is not None:
                stmt = stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)

            stmt = stmt.where(
                col(FOVAnalysis.global_spike_jitter_synchrony).is_not(None)
            )

            results = session.exec(stmt).all()

            data: dict[str, dict[str, float]] = {}
            for fov_analysis, fov, well in results:
                if fov_analysis.global_spike_jitter_synchrony is None:
                    continue
                cond_label = _get_condition_label(well)
                data.setdefault(cond_label, {})[fov.name] = (
                    fov_analysis.global_spike_jitter_synchrony
                )

        return data
    except OperationalError:
        return {}


def _query_spike_correlation_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, float]]:
    """Query mean spike correlation per FOV, grouped by condition.

    Uses pre-computed spike_correlation_matrix from FOVAnalysis.
    Returns the mean of off-diagonal elements as the global correlation metric.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {condition: {fov_name: mean_correlation_value}}
    """
    from sqlalchemy.exc import OperationalError
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, FOVAnalysis, Well

    try:
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis, FOV, Well)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .join(Well, FOV.well_id == Well.id)
            )

            if run_id is not None:
                stmt = stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)

            stmt = stmt.where(
                col(FOVAnalysis.spike_max_lag_correlation_matrix).is_not(None)
            )

            results = session.exec(stmt).all()

            data: dict[str, dict[str, float]] = {}
            for fov_analysis, fov, well in results:
                if fov_analysis.spike_max_lag_correlation_matrix is None:
                    continue

                corr_matrix = np.asarray(
                    fov_analysis.spike_max_lag_correlation_matrix, dtype=float
                )
                n = corr_matrix.shape[0]
                if n < 2:
                    continue

                mask = ~np.eye(n, dtype=bool)
                mean_corr = float(np.mean(corr_matrix[mask]))

                cond_label = _get_condition_label(well)
                data.setdefault(cond_label, {})[fov.name] = mean_corr

        return data
    except OperationalError:
        return {}


def plot_spike_synchrony_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes global synchrony across conditions."""
    data_by_condition = _query_spike_synchrony_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    data_as_lists: dict[str, dict[str, list[float]]] = {
        cond: {fov: [val] for fov, val in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
    }

    plot_data = _aggregate_fov_data_to_condition_stats(data_as_lists)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="Median",
        title_suffix=" (Median - Thresholded Data)",
        bar_label="Mean ± SEM (per FOV)",
    )


def plot_spike_correlation_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes global correlation across conditions.

    Uses the mean of off-diagonal correlation values from the spike
    correlation matrix stored in FOVAnalysis.
    """
    data_by_condition = _query_spike_correlation_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    data_as_lists: dict[str, dict[str, list[float]]] = {
        cond: {fov: [val] for fov, val in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
    }

    plot_data = _aggregate_fov_data_to_condition_stats(data_as_lists)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="Correlation",
        title_suffix=" (Mean Off-Diagonal)",
        bar_label="Mean ± SEM (per FOV)",
    )
