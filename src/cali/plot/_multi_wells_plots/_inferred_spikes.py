"""Inferred spikes and burst bar plots for multi-well analysis.

This module provides bar plot visualizations for inferred spike and burst metrics:
- Inferred spike frequency (thresholded spikes and rising edges)
- Burst count, average duration, average interval, and rate
- Spike synchrony and correlation across conditions
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ._util import (
    BarPlotData,
    _aggregate_fov_scalar_to_condition_stats,
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
        logging.getLogger(__name__).debug(
            "Failed to query spike burst metrics (table may not exist yet)",
            exc_info=True,
        )
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

    # Each FOV contributes a single scalar → use between-FOV SEM (weight=1)
    scalar_data: dict[str, dict[str, tuple[float, int]]] = {
        cond: {fov: (m[metric_key], 1) for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
    }

    plot_data = _aggregate_fov_scalar_to_condition_stats(scalar_data)

    if not plot_data["conditions"]:  # pragma: no cover
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
# Headless compute functions for CSV export
# ---------------------------------------------------------------------------


def _compute_burst_metric(
    engine: Engine,
    run_id: int | None,
    metric_key: str,
    name: str,
    units: str,
) -> tuple[BarPlotData, str, str] | None:
    """Compute a single burst metric without rendering."""
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)
    if not data_by_condition:
        return None
    scalar_data: dict[str, dict[str, tuple[float, int]]] = {
        cond: {fov: (m[metric_key], 1) for fov, m in fov_dict.items()}
        for cond, fov_dict in data_by_condition.items()
    }
    plot_data = _aggregate_fov_scalar_to_condition_stats(scalar_data)
    if not plot_data["conditions"]:
        return None
    return plot_data, name, units


def compute_burst_count_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst count data without rendering."""
    return _compute_burst_metric(engine, run_id, "count", "Burst Count", "Count")


def compute_burst_avg_duration_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst average duration data without rendering."""
    return _compute_burst_metric(
        engine, run_id, "avg_duration_sec", "Burst Average Duration", "s"
    )


def compute_burst_avg_interval_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst average interval data without rendering."""
    return _compute_burst_metric(
        engine, run_id, "avg_interval_sec", "Burst Average Interval", "s"
    )


def compute_burst_rate_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst rate data without rendering."""
    return _compute_burst_metric(
        engine, run_id, "rate_per_min", "Burst Rate", "bursts/min"
    )


# ---------------------------------------------------------------------------
# Spike synchrony and correlation bar plots
# ---------------------------------------------------------------------------


def _query_fov_scalar_by_condition(
    engine: Engine,
    run_id: int | None,
    field_name: str,
    *,
    use_n_pairs_weight: bool = True,
) -> dict[str, dict[str, tuple[float, int]]]:
    """Query a scalar FOVAnalysis field per FOV, grouped by condition.

    Parameters
    ----------
    engine : Engine
        Database engine.
    run_id : int | None
        Filter by specific analysis run.
    field_name : str
        Name of the ``FOVAnalysis`` attribute to read (must be a float field).
    use_n_pairs_weight : bool
        If True, weight is ``n_rois*(n_rois-1)//2`` (number of unique pairs).
        If False, weight is 1 (equal weight per FOV).

    Returns
    -------
    dict[str, dict[str, tuple[float, int]]]
        ``{condition: {fov_name: (scalar_value, weight)}}``
    """
    from sqlalchemy.exc import OperationalError
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, FOVAnalysis, Well

    try:
        with Session(engine) as session:
            field_col = getattr(FOVAnalysis, field_name)
            stmt = (
                select(FOVAnalysis, FOV, Well)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .join(Well, FOV.well_id == Well.id)
                .where(col(field_col).is_not(None))
            )

            if run_id is not None:
                stmt = stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)

            results = session.exec(stmt).all()

            data: dict[str, dict[str, tuple[float, int]]] = {}
            for fov_analysis, fov, well in results:
                value = getattr(fov_analysis, field_name)
                if value is None:
                    continue  # pragma: no cover

                if use_n_pairs_weight and fov_analysis.active_roi_labels:
                    n_rois = len(fov_analysis.active_roi_labels)
                    weight = max(1, n_rois * (n_rois - 1) // 2)
                else:
                    weight = 1

                cond_label = _get_condition_label(well)
                data.setdefault(cond_label, {})[fov.name] = (float(value), weight)

        return data
    except OperationalError:
        logging.getLogger(__name__).debug(
            "Failed to query %s (table may not exist yet)",
            field_name,
            exc_info=True,
        )
        return {}


def _plot_fov_scalar_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None,
    field_name: str,
    units: str,
    title_suffix: str = "",
    *,
    use_n_pairs_weight: bool = True,
) -> None:
    """Plot a FOV-level scalar metric across conditions."""
    data_by_condition = _query_fov_scalar_by_condition(
        engine, run_id, field_name, use_n_pairs_weight=use_n_pairs_weight
    )
    if not data_by_condition:
        widget.clear_plot()
        return

    plot_data = _aggregate_fov_scalar_to_condition_stats(data_by_condition)

    if not plot_data["conditions"]:  # pragma: no cover
        widget.clear_plot()
        return

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units=units,
        title_suffix=title_suffix,
        bar_label="Mean ± SEM (per FOV)",
    )


def _compute_fov_scalar_data(
    engine: Engine,
    run_id: int | None,
    field_name: str,
    name: str,
    units: str,
    *,
    use_n_pairs_weight: bool = True,
) -> tuple[BarPlotData, str, str] | None:
    """Compute a FOV-level scalar metric without rendering (for CSV export)."""
    data_by_condition = _query_fov_scalar_by_condition(
        engine, run_id, field_name, use_n_pairs_weight=use_n_pairs_weight
    )
    if not data_by_condition:
        return None
    plot_data = _aggregate_fov_scalar_to_condition_stats(data_by_condition)
    if not plot_data["conditions"]:
        return None
    return plot_data, name, units


def plot_spike_synchrony_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes global synchrony across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="global_spike_jitter_synchrony",
        units="Synchrony",
        title_suffix=" (Jitter Synchrony)",
    )


def compute_spike_synchrony_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike synchrony data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_jitter_synchrony",
        "Spike Jitter Synchrony",
        "Synchrony",
    )


def plot_spike_correlation_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes global max-lag correlation across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="global_spike_max_lag_correlation",
        units="Correlation",
        title_suffix=" (Max-Lag Cross-Correlation)",
    )


def compute_spike_correlation_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike correlation data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_max_lag_correlation",
        "Spike Max-Lag Correlation",
        "Correlation",
    )


# ---------------------------------------------------------------------------
# Calcium correlation bar plots
# ---------------------------------------------------------------------------


def plot_calcium_dff_correlation_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium ΔF/F correlation across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="global_calcium_dff_correlation",
        units="Correlation",
        title_suffix=" (Zero-Lag Pearson)",
    )


def compute_calcium_dff_correlation_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute calcium ΔF/F correlation data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_calcium_dff_correlation",
        "Calcium ΔF/F Correlation",
        "Correlation",
    )


def plot_calcium_den_dff_correlation_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium denoised ΔF/F correlation across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="global_calcium_den_dff_correlation",
        units="Correlation",
        title_suffix=" (Zero-Lag Pearson, Denoised)",
    )


def compute_calcium_den_dff_correlation_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute calcium denoised ΔF/F correlation data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_calcium_den_dff_correlation",
        "Calcium Denoised ΔF/F Correlation",
        "Correlation",
    )


# ---------------------------------------------------------------------------
# Rising edges bar plots
# ---------------------------------------------------------------------------


def plot_spike_synchrony_rising_edges_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot spike jitter synchrony (rising edges) across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="global_spike_jitter_synchrony_rising_edges",
        units="Synchrony",
        title_suffix=" (Jitter Synchrony, Rising Edges)",
    )


def compute_spike_synchrony_rising_edges_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike synchrony (rising edges) data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_jitter_synchrony_rising_edges",
        "Spike Jitter Synchrony (Rising Edges)",
        "Synchrony",
    )


def plot_spike_correlation_rising_edges_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot spike max-lag correlation (rising edges) across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="global_spike_max_lag_correlation_rising_edges",
        units="Correlation",
        title_suffix=" (Max-Lag Cross-Correlation, Rising Edges)",
    )


def compute_spike_correlation_rising_edges_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike correlation (rising edges) data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_max_lag_correlation_rising_edges",
        "Spike Max-Lag Correlation (Rising Edges)",
        "Correlation",
    )


# ---------------------------------------------------------------------------
# Fraction of significant CCG pairs bar plots
# ---------------------------------------------------------------------------


def plot_fraction_significant_ccg_pairs_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot fraction of significant CCG pairs across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="fraction_significant_ccg_pairs",
        units="Fraction",
        title_suffix=" (|z| > 2)",
    )


def compute_fraction_significant_ccg_pairs_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute fraction of significant CCG pairs data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "fraction_significant_ccg_pairs",
        "Fraction Significant CCG Pairs",
        "Fraction",
    )


def plot_fraction_significant_ccg_pairs_rising_edges_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot fraction of significant CCG pairs (rising edges) across conditions."""
    _plot_fov_scalar_bar_plot(
        widget,
        text,
        engine,
        run_id,
        field_name="fraction_significant_ccg_pairs_rising_edges",
        units="Fraction",
        title_suffix=" (|z| > 2, Rising Edges)",
    )


def compute_fraction_significant_ccg_pairs_rising_edges_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute fraction of significant CCG pairs (rising edges) data."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "fraction_significant_ccg_pairs_rising_edges",
        "Fraction Significant CCG Pairs (Rising Edges)",
        "Fraction",
    )
