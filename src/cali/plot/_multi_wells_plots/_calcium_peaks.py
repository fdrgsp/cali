"""Calcium peaks related bar plots for multi-well analysis.

This module provides bar plot visualizations for calcium peak metrics:
- Amplitude
- Frequency
- Inter-event interval (IEI)
- Calcium population burst count, average duration, and average interval
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from ._util import (
    BarPlotData,
    _aggregate_fov_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
    plot_parameter_bar_plot,
)

if TYPE_CHECKING:
    import numpy as np
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


def plot_calcium_peaks_amplitude_stim_split_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks amplitude split by stim/non-stim with LED power labels.

    For evoked experiments.  Stim bars are green and labelled with the LED power
    percentage (e.g. `"ctrl (25%)"`); non-stim bars are magenta with the same
    power labels.  Both sets are shown in a single combined plot.
    """
    from cali._constants import EVK_NON_STIM, EVK_STIM

    from ._evoked_activity import (
        _aggregate_evoked_data_to_condition_stats,
        _query_evoked_amplitudes_by_condition,
    )

    # Query stim and non-stim amplitudes
    stim_data = _query_evoked_amplitudes_by_condition(
        engine, stimulated=True, run_id=run_id
    )
    non_stim_data = _query_evoked_amplitudes_by_condition(
        engine, stimulated=False, run_id=run_id
    )

    if not stim_data and not non_stim_data:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # Aggregate each side; produces conditions like "ctrl (25%)"
    stim_plot_data = (
        _aggregate_evoked_data_to_condition_stats(stim_data) if stim_data else None
    )
    non_stim_plot_data = (
        _aggregate_evoked_data_to_condition_stats(non_stim_data)
        if non_stim_data
        else None
    )

    # Build per-condition lookups keyed by the base condition name (e.g. "ctrl (25%)")
    stim_lookup: dict[str, tuple[float, float, np.ndarray]] = {}
    if stim_plot_data:
        for cond, mean, sem, fov_vals in zip(
            stim_plot_data["conditions"],
            stim_plot_data["means"],
            stim_plot_data["sems"],
            stim_plot_data["fov_values_list"],
        ):
            stim_lookup[cond] = (mean, sem, fov_vals)

    non_stim_lookup: dict[str, tuple[float, float, np.ndarray]] = {}
    if non_stim_plot_data:
        for cond, mean, sem, fov_vals in zip(
            non_stim_plot_data["conditions"],
            non_stim_plot_data["means"],
            non_stim_plot_data["sems"],
            non_stim_plot_data["fov_values_list"],
        ):
            non_stim_lookup[cond] = (mean, sem, fov_vals)

    # Collect all unique base conditions and sort by (base_name, numeric_power)
    # so that the x-axis order is: ctrl (25%) stim, ctrl (25%) non-stim,
    # ctrl (50%) stim, ctrl (50%) non-stim, trt (25%) stim, trt (25%) non-stim, …
    seen: set[str] = set()
    all_base_conditions: list[str] = []
    for cond in list(stim_lookup) + list(non_stim_lookup):
        if cond not in seen:
            seen.add(cond)
            all_base_conditions.append(cond)

    def _sort_key(c: str) -> tuple[str, float]:
        base = c.rsplit(" (", 1)[0]
        tail = c.rsplit(" (", 1)[-1] if " (" in c else ""
        m = re.search(r"(\d+\.?\d*)", tail)
        return (base, float(m.group(1)) if m else 0.0)

    all_base_conditions.sort(key=_sort_key)

    # Interleave: for each condition+power, stim bar then non-stim bar
    combined_conditions: list[str] = []
    combined_means: list[float] = []
    combined_sems: list[float] = []
    combined_fov_values: list[np.ndarray] = []

    for base_cond in all_base_conditions:
        if base_cond in stim_lookup:
            mean, sem, fov_vals = stim_lookup[base_cond]
            combined_conditions.append(f"{base_cond}_{EVK_STIM}")
            combined_means.append(mean)
            combined_sems.append(sem)
            combined_fov_values.append(fov_vals)
        if base_cond in non_stim_lookup:
            mean, sem, fov_vals = non_stim_lookup[base_cond]
            combined_conditions.append(f"{base_cond}_{EVK_NON_STIM}")
            combined_means.append(mean)
            combined_sems.append(sem)
            combined_fov_values.append(fov_vals)

    if not combined_conditions:  # pragma: no cover
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    combined_plot_data: BarPlotData = {
        "conditions": combined_conditions,
        "means": combined_means,
        "sems": combined_sems,
        "fov_values_list": combined_fov_values,
    }

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=combined_plot_data,
        parameter=text,
        units="ΔF/F0",
        title_suffix="",
        bar_label="Mean ± SEM (per FOV)",
    )


def plot_calcium_peaks_frequency_stim_split_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks frequency split by stim/non-stim within each condition.

    Evoked-only: condition labels are suffixed with '(Stim)' or '(NonStim)'.
    """
    plot_parameter_bar_plot(
        widget,
        text,
        engine,
        run_id,
        parameter="den_dff_frequency",
        units="Hz",
        include_stim_status=True,
    )


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
                if not fa.calcium_burst_count:  # skip None and 0 (no bursts detected)
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
        logging.getLogger(__name__).debug(
            "Failed to query calcium burst metrics (table may not exist yet)",
            exc_info=True,
        )
        return {}


def _plot_calcium_burst_metric(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None,
    metric_key: str,
    units: str,
) -> None:
    """Plot a single calcium burst metric across conditions.

    NOTE:
    metric_key : Key in the burst metrics dict (e.g. `"count"`, `"avg_duration_s"`).
    units : Y-axis units label.
    """
    data_by_condition = _query_calcium_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    metric_data: dict[str, dict[str, list[float]]] = {
        cond: {fov: [m[metric_key]] for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
    }

    plot_data = _aggregate_fov_data_to_condition_stats(metric_data)
    if not plot_data["conditions"]:  # pragma: no cover
        widget.clear_plot()
        return

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units=units,
        title_suffix=" (Calcium Peaks)",
        bar_label="Mean ± SEM (per FOV)",
    )


def plot_calcium_burst_count_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium population burst count across conditions."""
    _plot_calcium_burst_metric(widget, text, engine, run_id, "count", "Count")


def plot_calcium_burst_avg_duration_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium population burst average duration across conditions."""
    _plot_calcium_burst_metric(widget, text, engine, run_id, "avg_duration_s", "s")


def plot_calcium_burst_avg_interval_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium population burst average interval across conditions."""
    _plot_calcium_burst_metric(widget, text, engine, run_id, "avg_interval_s", "s")


# ---------------------------------------------------------------------------
# Headless compute functions for CSV export
# ---------------------------------------------------------------------------


def _compute_evoked_stim_split_data(
    engine: Engine,
    run_id: int | None,
    name: str,
    units: str,
) -> tuple[BarPlotData, str, str] | None:
    """Compute evoked stim-split bar plot data without rendering."""
    from cali._constants import EVK_NON_STIM, EVK_STIM

    from ._evoked_activity import (
        _aggregate_evoked_data_to_condition_stats,
        _query_evoked_amplitudes_by_condition,
    )

    stim_data = _query_evoked_amplitudes_by_condition(
        engine, stimulated=True, run_id=run_id
    )
    non_stim_data = _query_evoked_amplitudes_by_condition(
        engine, stimulated=False, run_id=run_id
    )
    if not stim_data and not non_stim_data:
        return None

    stim_plot_data = (
        _aggregate_evoked_data_to_condition_stats(stim_data) if stim_data else None
    )
    non_stim_plot_data = (
        _aggregate_evoked_data_to_condition_stats(non_stim_data)
        if non_stim_data
        else None
    )

    stim_lookup: dict[str, tuple[float, float, np.ndarray]] = {}
    if stim_plot_data:
        for cond, mean, sem, fov_vals in zip(
            stim_plot_data["conditions"],
            stim_plot_data["means"],
            stim_plot_data["sems"],
            stim_plot_data["fov_values_list"],
        ):
            stim_lookup[cond] = (mean, sem, fov_vals)

    non_stim_lookup: dict[str, tuple[float, float, np.ndarray]] = {}
    if non_stim_plot_data:
        for cond, mean, sem, fov_vals in zip(
            non_stim_plot_data["conditions"],
            non_stim_plot_data["means"],
            non_stim_plot_data["sems"],
            non_stim_plot_data["fov_values_list"],
        ):
            non_stim_lookup[cond] = (mean, sem, fov_vals)

    seen: set[str] = set()
    all_base_conditions: list[str] = []
    for cond in list(stim_lookup) + list(non_stim_lookup):
        if cond not in seen:
            seen.add(cond)
            all_base_conditions.append(cond)

    def _sort_key(c: str) -> tuple[str, float]:
        base = c.rsplit(" (", 1)[0]
        tail = c.rsplit(" (", 1)[-1] if " (" in c else ""
        m = re.search(r"(\d+\.?\d*)", tail)
        return (base, float(m.group(1)) if m else 0.0)

    all_base_conditions.sort(key=_sort_key)

    combined_conditions: list[str] = []
    combined_means: list[float] = []
    combined_sems: list[float] = []
    combined_fov_values: list[np.ndarray] = []

    for base_cond in all_base_conditions:
        if base_cond in stim_lookup:
            mean, sem, fov_vals = stim_lookup[base_cond]
            combined_conditions.append(f"{base_cond}_{EVK_STIM}")
            combined_means.append(mean)
            combined_sems.append(sem)
            combined_fov_values.append(fov_vals)
        if base_cond in non_stim_lookup:
            mean, sem, fov_vals = non_stim_lookup[base_cond]
            combined_conditions.append(f"{base_cond}_{EVK_NON_STIM}")
            combined_means.append(mean)
            combined_sems.append(sem)
            combined_fov_values.append(fov_vals)

    if not combined_conditions:
        return None

    return (
        BarPlotData(
            conditions=combined_conditions,
            means=combined_means,
            sems=combined_sems,
            fov_values_list=combined_fov_values,
        ),
        name,
        units,
    )


def compute_calcium_amplitude_stim_split_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute calcium amplitude stim-split data without rendering."""
    return _compute_evoked_stim_split_data(
        engine, run_id, "Calcium Peaks Amplitude (Stim vs NonStim)", "ΔF/F0"
    )


def _compute_calcium_burst_metric(
    engine: Engine,
    run_id: int | None,
    metric_key: str,
    name: str,
    units: str,
) -> tuple[BarPlotData, str, str] | None:
    """Compute a single calcium burst metric without rendering."""
    data_by_condition = _query_calcium_burst_metrics_by_condition(engine, run_id)
    if not data_by_condition:
        return None
    metric_data: dict[str, dict[str, list[float]]] = {
        cond: {fov: [m[metric_key]] for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
    }
    plot_data = _aggregate_fov_data_to_condition_stats(metric_data)
    if not plot_data["conditions"]:
        return None
    return plot_data, name, units


def compute_calcium_burst_count_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute calcium burst count data without rendering."""
    return _compute_calcium_burst_metric(
        engine, run_id, "count", "Calcium Burst Count", "Count"
    )


def compute_calcium_burst_avg_duration_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute calcium burst average duration data without rendering."""
    return _compute_calcium_burst_metric(
        engine, run_id, "avg_duration_s", "Calcium Burst Average Duration", "s"
    )


def compute_calcium_burst_avg_interval_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute calcium burst average interval data without rendering."""
    return _compute_calcium_burst_metric(
        engine, run_id, "avg_interval_s", "Calcium Burst Average Interval", "s"
    )
