"""Calcium peaks related bar plots for multi-well analysis.

This module provides bar plot visualizations for calcium peak metrics:
- Amplitude
- Frequency
- Inter-event interval (IEI)
- Synchrony
- Correlation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.exc import OperationalError
from sqlmodel import Session, col, select

from cali.sqlmodel import FOV, FOVAnalysis, Well

from ._util import (
    _aggregate_fov_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
    plot_parameter_bar_plot,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


def _query_calcium_peaks_correlation_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, float]]:
    """Query mean calcium peaks correlation per FOV, grouped by condition.

    Uses pre-computed calcium_peaks_max_lag_correlation_matrix from FOVAnalysis.
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
    import numpy as np

    try:
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis, FOV, Well)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .join(Well, FOV.well_id == Well.id)
            )

            if run_id is not None:
                stmt = stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)

            # Only include FOVs with valid correlation data
            stmt = stmt.where(
                col(FOVAnalysis.calcium_dff_correlation_matrix).is_not(None)
            )

            results = session.exec(stmt).all()

            data: dict[str, dict[str, float]] = {}
            for fov_analysis, fov, well in results:
                if fov_analysis.calcium_peaks_max_lag_correlation_matrix is None:
                    continue

                # Calculate mean of off-diagonal correlation values
                corr_matrix = np.asarray(
                    fov_analysis.calcium_peaks_max_lag_correlation_matrix, dtype=float
                )
                n = corr_matrix.shape[0]
                if n < 2:
                    continue

                # Mask out diagonal
                mask = ~np.eye(n, dtype=bool)
                mean_corr = float(np.mean(corr_matrix[mask]))

                cond_label = _get_condition_label(well)
                data.setdefault(cond_label, {})[fov.name] = mean_corr

        return data
    except OperationalError:
        # Table doesn't exist in older databases
        return {}


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
        parameter="peaks_amplitudes_dec_dff",
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
        widget, text, engine, run_id, parameter="dec_dff_frequency", units="Hz"
    )


def plot_calcium_peaks_iei_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks inter-event interval across conditions."""
    plot_parameter_bar_plot(widget, text, engine, run_id, parameter="iei", units="s")


def plot_calcium_peaks_correlation_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peak events global correlation across conditions."""
    # Query correlation data (one value per FOV)
    data_by_condition = _query_calcium_peaks_correlation_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Convert to format expected by aggregation
    data_as_lists: dict[str, dict[str, list[float]]] = {}
    for condition, fov_dict in data_by_condition.items():
        for fov_name, corr_value in fov_dict.items():
            data_as_lists.setdefault(condition, {})[fov_name] = [corr_value]

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
        title_suffix=" (Mean)",
        bar_label="Weighted Mean ± Pooled SEM",
    )
