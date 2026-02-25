"""Inferred spikes and burst bar plots for multi-well analysis.

This module provides bar plot visualizations for inferred spike and burst metrics:
- Inferred spike frequency (thresholded spikes and rising edges)
- Burst count, average duration, average interval, and rate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    Uses stored ``FOVAnalysis.spike_burst_count``, ``spike_burst_avg_duration``,
    and ``spike_burst_avg_interval`` — the same values highlighted in the
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
        title_suffix=" (Inferred Spikes)",
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
        title_suffix=" (Inferred Spikes)",
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
        title_suffix=" (Inferred Spikes)",
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
        title_suffix=" (Inferred Spikes)",
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
