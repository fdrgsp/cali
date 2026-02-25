"""Calcium peaks related bar plots for multi-well analysis.

This module provides bar plot visualizations for calcium peak metrics:
- Amplitude
- Frequency
- Inter-event interval (IEI)
- Calcium population burst count, average duration, and average interval
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


def plot_calcium_peaks_amplitude_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks amplitude across conditions."""
    plot_parameter_bar_plot(
        widget,
        text,
        engine,
        run_id,
        parameter="peaks_amplitudes_den_dff",
        units="ΔF/F0",
    )


def plot_calcium_peaks_frequency_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks frequency across conditions."""
    plot_parameter_bar_plot(
        widget, text, engine, run_id, parameter="den_dff_frequency", units="Hz"
    )


def plot_calcium_peaks_iei_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks inter-event interval across conditions."""
    plot_parameter_bar_plot(widget, text, engine, run_id, parameter="iei", units="s")


def _query_calcium_burst_metrics_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Query pre-computed calcium burst metrics from FOVAnalysis, grouped by condition.

    Parameters
    ----------
    engine : Engine
        Database engine.
    run_id : int | None
        Filter by specific analysis run.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Nested dict: {condition: {fov_name: {"count": ..., "avg_duration_s": ...,
        "avg_interval_s": ...}}}
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

            results = session.exec(stmt).all()

            data: dict[str, dict[str, dict[str, float]]] = {}
            for fa, fov, well in results:
                if fa.calcium_burst_count is None:
                    continue
                cond_label = _get_condition_label(well)
                data.setdefault(cond_label, {})[fov.name] = {
                    "count": float(fa.calcium_burst_count),
                    "avg_duration_s": (
                        float(fa.calcium_burst_avg_duration)
                        if fa.calcium_burst_avg_duration is not None
                        else 0.0
                    ),
                    "avg_interval_s": (
                        float(fa.calcium_burst_avg_interval)
                        if fa.calcium_burst_avg_interval is not None
                        else 0.0
                    ),
                }

        return data
    except OperationalError:
        return {}


def plot_calcium_burst_count_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium population burst count across conditions."""
    data_by_condition = _query_calcium_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    count_data: dict[str, dict[str, list[float]]] = {
        cond: {fov: [m["count"]] for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
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
        title_suffix="(Calcium Peaks)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_calcium_burst_avg_duration_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium population burst average duration across conditions."""
    data_by_condition = _query_calcium_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    duration_data: dict[str, dict[str, list[float]]] = {
        cond: {fov: [m["avg_duration_s"]] for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
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
        title_suffix="(Calcium Peaks)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_calcium_burst_avg_interval_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium population burst average interval across conditions."""
    data_by_condition = _query_calcium_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    interval_data: dict[str, dict[str, list[float]]] = {
        cond: {fov: [m["avg_interval_s"]] for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
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
        title_suffix="(Calcium Peaks)",
        bar_label="Weighted Mean ± Pooled SEM",
    )
