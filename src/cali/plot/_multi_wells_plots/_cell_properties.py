"""Cell properties bar plots for multi-well analysis.

This module provides bar plot visualizations for cell properties:
- Cell size
- Percentage active ROIs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import Session, col, select

from cali.sqlmodel import FOV, ROI, AnalysisSettings, DataAnalysis, Well

from ._util import (
    _aggregate_fov_data_to_condition_stats,
    _aggregate_percentage_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
    _query_roi_attribute_by_condition,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


def _query_fov_percentage_active(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, tuple[float, int]]]:
    """Query percentage of active ROIs per FOV, grouped by condition.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, tuple[float, int]]]
        Nested dict: {condition_label: {fov_name: (percentage, n_total)}}
    """
    with Session(engine) as session:
        # Get experiment type if run_id is provided
        experiment_type = None
        if run_id is not None:
            from cali.sqlmodel import CaliResult

            stmt_exp_type = (
                select(AnalysisSettings.experiment_type)
                .join(
                    CaliResult, CaliResult.analysis_settings_id == AnalysisSettings.id
                )
                .where(CaliResult.id == run_id)
            )
            experiment_type = session.exec(stmt_exp_type).first()

        # Get all ROIs grouped by FOV - start from ROI and join backwards
        stmt = (
            select(ROI, FOV, Well)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
        )

        if run_id is not None:
            # Filter by ROIs that have DataAnalysis for this run
            stmt = stmt.join(DataAnalysis, DataAnalysis.roi_id == ROI.id).where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        results = session.exec(stmt).all()

        # Group by condition and FOV, count active vs total
        data: dict[str, dict[str, tuple[int, int]]] = {}
        for roi, fov, well in results:
            # Get condition label (including stimulation status for evoked exps)
            cond_label = _get_condition_label(well, roi, experiment_type)

            if cond_label not in data:
                data[cond_label] = {}
            if fov.name not in data[cond_label]:
                data[cond_label][fov.name] = (0, 0)

            active_count, total_count = data[cond_label][fov.name]
            total_count += 1
            if roi.active:
                active_count += 1
            data[cond_label][fov.name] = (active_count, total_count)

        # Convert to percentages
        result: dict[str, dict[str, tuple[float, int]]] = {}
        for cond_label, fov_dict in data.items():
            result[cond_label] = {}
            for fov_name, (active, total) in fov_dict.items():
                percentage = (active / total * 100) if total > 0 else 0.0
                result[cond_label][fov_name] = (percentage, total)

    return result


def plot_cell_size_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot cell size across conditions.

    Cell size is stored in DataAnalysis (versioned per run).
    """
    # Query data using ROI attribute query
    data_by_condition = _query_roi_attribute_by_condition(
        engine, attribute="cell_size", run_id=run_id
    )

    if not data_by_condition:
        widget.clear_plot()
        return

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(data_by_condition)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="μm²",
        title_suffix="",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_percentage_active_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot percentage of active ROIs per condition."""
    # Query percentage active data
    data_by_condition = _query_fov_percentage_active(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Aggregate to condition-level statistics
    plot_data = _aggregate_percentage_data_to_condition_stats(data_by_condition)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter="Percentage Active ROIs",
        units="%",
        title_suffix="",
        bar_label="Weighted Mean ± Binomial SEM",
    )
