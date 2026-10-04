"""Inferred spikes and burst bar plots for multi-well analysis.

This module provides bar plot visualizations for inferred spike and burst metrics:
- Inferred spike frequency (thresholded spikes and rising edges)
- Burst count, average duration, average interval, and rate
- Spike synchrony and correlation across conditions
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cali.plot._spike_fov_data import selected_spike_fov, spike_population_duration
from cali.sqlmodel._engine import ensure_schema_current
from cali.sqlmodel._spike_settings import canonical_spike_methods

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
    from cali.sqlmodel._spike_settings import SpikeMethod


def _query_burst_metrics_by_condition(
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Read one method's population, preserving zero counts and unknown metrics."""
    from sqlalchemy.exc import OperationalError
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, FOVAnalysis, Well
    from cali.sqlmodel._model import AnalysisSettings, CaliResult

    canonical_spike_methods((spike_method,))
    try:
        ensure_schema_current(engine)
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis, FOV, Well, AnalysisSettings)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .join(Well, FOV.well_id == Well.id)
                .join(CaliResult, FOVAnalysis.analysis_result_id == CaliResult.id)
                .outerjoin(
                    AnalysisSettings,
                    col(CaliResult.analysis_settings_id) == col(AnalysisSettings.id),
                )
            )
            if run_id is not None:
                stmt = stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)
            data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
            for parent, fov, well, settings in session.exec(stmt).all():
                child = selected_spike_fov(parent, spike_method)
                if child is None or child.spike_burst_count is None:
                    continue
                metrics = {"count": float(child.spike_burst_count)}
                for key, value in (
                    ("avg_duration_sec", child.spike_burst_avg_duration),
                    ("avg_interval_sec", child.spike_burst_avg_interval),
                ):
                    if value is not None:
                        metrics[key] = float(value)
                duration = spike_population_duration(child, settings)
                if duration is not None and duration > 0:
                    metrics["rate_per_min"] = child.spike_burst_count * 60 / duration
                condition = _get_condition_label(well)
                data.setdefault(condition, {}).setdefault(well.name, {})[fov.name] = (
                    metrics
                )
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
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot a single burst metric across conditions.

    NOTE:
    metric_key: Key in the burst metrics dict (e.g. `"count"`, `"avg_duration_sec"`).
    units: Y-axis units label.
    """
    data_by_condition = _query_burst_metrics_by_condition(
        engine, run_id, spike_method=spike_method
    )

    if not data_by_condition:
        widget.clear_plot()
        return

    # Each FOV contributes a single scalar → use between-well SEM (weight=1)
    scalar_data: dict[str, dict[str, dict[str, tuple[float, int]]]] = {
        cond: {
            well: {
                fov: (m[metric_key], 1)
                for fov, m in fov_dict.items()
                if metric_key in m
            }
            for well, fov_dict in well_dict.items()
        }
        for cond, well_dict in data_by_condition.items()
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
        title_suffix=f" ({spike_method.upper()} Inferred Spikes)",
        bar_label="Mean ± SEM (per FOV)",
    )


def plot_burst_count_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot burst count across conditions."""
    _plot_burst_metric(
        widget, text, engine, run_id, "count", "Count", spike_method=spike_method
    )


def plot_burst_avg_duration_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot burst average duration across conditions."""
    _plot_burst_metric(
        widget, text, engine, run_id, "avg_duration_sec", "s", spike_method=spike_method
    )


def plot_burst_avg_interval_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot burst average interval across conditions."""
    _plot_burst_metric(
        widget, text, engine, run_id, "avg_interval_sec", "s", spike_method=spike_method
    )


def plot_burst_rate_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot burst rate across conditions."""
    _plot_burst_metric(
        widget,
        text,
        engine,
        run_id,
        "rate_per_min",
        "bursts/min",
        spike_method=spike_method,
    )


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
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute a single burst metric without rendering."""
    data_by_condition = _query_burst_metrics_by_condition(
        engine, run_id, spike_method=spike_method
    )
    if not data_by_condition:
        return None
    scalar_data: dict[str, dict[str, dict[str, tuple[float, int]]]] = {
        cond: {
            well: {
                fov: (m[metric_key], 1)
                for fov, m in fov_dict.items()
                if metric_key in m
            }
            for well, fov_dict in well_dict.items()
        }
        for cond, well_dict in data_by_condition.items()
    }
    plot_data = _aggregate_fov_scalar_to_condition_stats(scalar_data)
    if not plot_data["conditions"]:
        return None
    if spike_method == "cascade":
        name = "CASCADE " + name.replace("Rising Edges", "Threshold Excursion Starts")
    return plot_data, name, units


def compute_burst_count_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst count data without rendering."""
    return _compute_burst_metric(
        engine, run_id, "count", "Burst Count", "Count", spike_method=spike_method
    )


def compute_burst_avg_duration_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst average duration data without rendering."""
    return _compute_burst_metric(
        engine,
        run_id,
        "avg_duration_sec",
        "Burst Average Duration",
        "s",
        spike_method=spike_method,
    )


def compute_burst_avg_interval_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst average interval data without rendering."""
    return _compute_burst_metric(
        engine,
        run_id,
        "avg_interval_sec",
        "Burst Average Interval",
        "s",
        spike_method=spike_method,
    )


def compute_burst_rate_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute burst rate data without rendering."""
    return _compute_burst_metric(
        engine,
        run_id,
        "rate_per_min",
        "Burst Rate",
        "bursts/min",
        spike_method=spike_method,
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
    spike_method: SpikeMethod = "oasis",
) -> dict[str, dict[str, dict[str, tuple[float, int]]]]:
    """Query a scalar FOVAnalysis field per FOV, grouped by condition and well.

    Parameters
    ----------
    engine : Engine
        Database engine.
    run_id : int | None
        Filter by specific analysis run.
    field_name : str
        Name of the ``FOVAnalysis`` attribute to read (must be a float field).
    spike_method : {"oasis", "cascade"}
        Requested stored population; calcium fields remain shared.
    use_n_pairs_weight : bool
        If True, weight is ``n_rois*(n_rois-1)//2`` (number of unique pairs).
        If False, weight is 1 (equal weight per FOV).

    Returns
    -------
    dict[str, dict[str, dict[str, tuple[float, int]]]]
        ``{condition: {well_id: {fov_name: (scalar_value, weight)}}}``
    """
    from sqlalchemy.exc import OperationalError
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, FOVAnalysis, SpikeFOVAnalysis, Well
    from cali.sqlmodel._spike_fov_analysis import SPIKE_FOV_METRICS

    canonical_spike_methods((spike_method,))
    try:
        ensure_schema_current(engine)
        with Session(engine) as session:
            is_spike_metric = field_name in SPIKE_FOV_METRICS
            field_col = getattr(
                SpikeFOVAnalysis if is_spike_metric else FOVAnalysis, field_name
            )
            stmt = (
                select(FOVAnalysis, FOV, Well)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .join(Well, FOV.well_id == Well.id)
            )
            if is_spike_metric:
                stmt = stmt.join(
                    SpikeFOVAnalysis,
                    col(SpikeFOVAnalysis.fov_analysis_id) == FOVAnalysis.id,
                ).where(
                    SpikeFOVAnalysis.method == spike_method, col(field_col).is_not(None)
                )
            else:
                stmt = stmt.where(col(field_col).is_not(None))

            if run_id is not None:
                stmt = stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)

            results = session.exec(stmt).all()

            data: dict[str, dict[str, dict[str, tuple[float, int]]]] = {}
            for fov_analysis, fov, well in results:
                child = (
                    selected_spike_fov(fov_analysis, spike_method)
                    if is_spike_metric
                    else None
                )
                if is_spike_metric and child is None:
                    continue
                value = (
                    getattr(child, field_name)
                    if is_spike_metric
                    else getattr(fov_analysis, field_name)
                )
                if value is None:
                    continue  # pragma: no cover

                roi_labels = (
                    child.active_roi_labels
                    if child is not None
                    else fov_analysis.calcium_active_roi_labels
                )
                if use_n_pairs_weight and roi_labels:
                    n_rois = len(roi_labels)
                    weight = max(1, n_rois * (n_rois - 1) // 2)
                else:
                    weight = 1

                cond_label = _get_condition_label(well)
                well_key = well.name
                data.setdefault(cond_label, {}).setdefault(well_key, {})[fov.name] = (
                    float(value),
                    weight,
                )

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
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot a FOV-level scalar metric across conditions."""
    data_by_condition = _query_fov_scalar_by_condition(
        engine,
        run_id,
        field_name,
        use_n_pairs_weight=use_n_pairs_weight,
        spike_method=spike_method,
    )
    if not data_by_condition:
        widget.clear_plot()
        return

    plot_data = _aggregate_fov_scalar_to_condition_stats(data_by_condition)

    if not plot_data["conditions"]:  # pragma: no cover
        widget.clear_plot()
        return

    if spike_method == "cascade":
        text = text.replace("Rising Edges", "Threshold Excursion Starts")
        title_suffix = title_suffix.replace(
            "Rising Edges", "Threshold Excursion Starts"
        )
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units=units,
        title_suffix=(
            f" [{spike_method.upper()}]" + title_suffix
            if "spike" in field_name or "ccg" in field_name
            else title_suffix
        ),
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
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute a FOV-level scalar metric without rendering (for CSV export)."""
    data_by_condition = _query_fov_scalar_by_condition(
        engine,
        run_id,
        field_name,
        use_n_pairs_weight=use_n_pairs_weight,
        spike_method=spike_method,
    )
    if not data_by_condition:
        return None
    plot_data = _aggregate_fov_scalar_to_condition_stats(data_by_condition)
    if not plot_data["conditions"]:
        return None
    if spike_method == "cascade":
        name = "CASCADE " + name.replace("Rising Edges", "Threshold Excursion Starts")
    return plot_data, name, units


def plot_spike_synchrony_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
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
        spike_method=spike_method,
    )


def compute_spike_synchrony_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike synchrony data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_jitter_synchrony",
        "Spike Jitter Synchrony",
        "Synchrony",
        spike_method=spike_method,
    )


def plot_spike_correlation_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
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
        spike_method=spike_method,
    )


def compute_spike_correlation_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike correlation data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_max_lag_correlation",
        "Spike Max-Lag Correlation",
        "Correlation",
        spike_method=spike_method,
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
    *,
    spike_method: SpikeMethod = "oasis",
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
        spike_method=spike_method,
    )


def compute_spike_synchrony_rising_edges_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike synchrony (rising edges) data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_jitter_synchrony_rising_edges",
        "Spike Jitter Synchrony (Rising Edges)",
        "Synchrony",
        spike_method=spike_method,
    )


def plot_spike_correlation_rising_edges_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
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
        spike_method=spike_method,
    )


def compute_spike_correlation_rising_edges_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute spike correlation (rising edges) data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "global_spike_max_lag_correlation_rising_edges",
        "Spike Max-Lag Correlation (Rising Edges)",
        "Correlation",
        spike_method=spike_method,
    )


# ---------------------------------------------------------------------------
# Fraction of significant CCG pairs bar plots
# ---------------------------------------------------------------------------


def plot_fraction_significant_ccg_pairs_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
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
        spike_method=spike_method,
    )


def compute_fraction_significant_ccg_pairs_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute fraction of significant CCG pairs data without rendering."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "fraction_significant_ccg_pairs",
        "Fraction Significant CCG Pairs",
        "Fraction",
        spike_method=spike_method,
    )


def plot_fraction_significant_ccg_pairs_rising_edges_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
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
        spike_method=spike_method,
    )


def compute_fraction_significant_ccg_pairs_rising_edges_data(
    engine: Engine,
    run_id: int | None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[BarPlotData, str, str] | None:
    """Compute fraction of significant CCG pairs (rising edges) data."""
    return _compute_fov_scalar_data(
        engine,
        run_id,
        "fraction_significant_ccg_pairs_rising_edges",
        "Fraction Significant CCG Pairs (Rising Edges)",
        "Fraction",
        spike_method=spike_method,
    )
