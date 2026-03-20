"""Cell properties bar plots for multi-well analysis.

This module provides bar plot visualizations for cell properties:
- Cell size
- Percentage active ROIs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import Session, col, select

from cali.sqlmodel import FOV, ROI, DataAnalysis, Well

from ._util import (
    BarPlotData,
    _aggregate_fov_data_to_condition_stats,
    _aggregate_percentage_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
    _get_experiment_type,
    _query_roi_attribute_by_condition,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


def _query_fov_percentage_active(
    engine: Engine,
    run_id: int | None = None,
    include_stim_status: bool = False,
) -> dict[str, dict[str, dict[str, tuple[float, int]]]]:
    """Query percentage of active ROIs per FOV, grouped by condition and well.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run
    include_stim_status : bool
        If True, condition labels include stim/non-stim split for evoked experiments.

    Returns
    -------
    dict[str, dict[str, dict[str, tuple[float, int]]]]
        Nested dict: {condition_label: {well_id: {fov_name: (percentage, n_total)}}}
    """
    with Session(engine) as session:
        # Get experiment type when stim split is requested
        experiment_type = None
        if include_stim_status and run_id is not None:
            experiment_type = _get_experiment_type(session, run_id)

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

        # Group by condition → well → FOV, count active vs total
        data: dict[str, dict[str, dict[str, tuple[int, int]]]] = {}
        for roi, fov, well in results:
            cond_label = (
                _get_condition_label(well, roi, experiment_type)
                if include_stim_status
                else _get_condition_label(well)
            )
            well_key = well.name
            if cond_label not in data:
                data[cond_label] = {}
            if well_key not in data[cond_label]:
                data[cond_label][well_key] = {}
            if fov.name not in data[cond_label][well_key]:
                data[cond_label][well_key][fov.name] = (0, 0)

            active_count, total_count = data[cond_label][well_key][fov.name]
            total_count += 1
            if roi.active:
                active_count += 1
            data[cond_label][well_key][fov.name] = (active_count, total_count)

        # Convert to percentages
        result: dict[str, dict[str, dict[str, tuple[float, int]]]] = {}
        for cond_label, well_dict in data.items():
            result[cond_label] = {}
            for well_key, fov_dict in well_dict.items():
                result[cond_label][well_key] = {}
                for fov_name, (active, total) in fov_dict.items():
                    percentage = (active / total * 100) if total > 0 else 0.0
                    result[cond_label][well_key][fov_name] = (percentage, total)

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
        bar_label="Mean ± SEM (per FOV)",
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
        bar_label="Mean ± SEM (per FOV)",
    )


def plot_percentage_active_stim_split_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot percentage of active ROIs split by stim/non-stim within each condition.

    Evoked-only: condition labels are suffixed with '(Stim)' or '(NonStim)'.
    """
    data_by_condition = _query_fov_percentage_active(
        engine, run_id, include_stim_status=True
    )

    if not data_by_condition:
        widget.clear_plot()
        return

    plot_data = _aggregate_percentage_data_to_condition_stats(data_by_condition)

    if not plot_data["conditions"]:  # pragma: no cover
        widget.clear_plot()
        return

    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter="Percentage Active ROIs",
        units="%",
        title_suffix="",
        bar_label="Mean ± SEM (per FOV)",
    )


# ---------------------------------------------------------------------------
# Headless compute functions for CSV export
# ---------------------------------------------------------------------------


def compute_cell_size_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute cell size bar plot data without rendering."""
    data_by_condition = _query_roi_attribute_by_condition(
        engine, attribute="cell_size", run_id=run_id
    )
    if not data_by_condition:
        return None
    plot_data = _aggregate_fov_data_to_condition_stats(data_by_condition)
    if not plot_data["conditions"]:
        return None
    return plot_data, "Cell Size", "μm²"


def compute_percentage_active_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute percentage active bar plot data without rendering."""
    data_by_condition = _query_fov_percentage_active(engine, run_id)
    if not data_by_condition:
        return None
    plot_data = _aggregate_percentage_data_to_condition_stats(data_by_condition)
    if not plot_data["conditions"]:
        return None
    return plot_data, "Percentage Active ROIs", "%"


def compute_percentage_active_stim_split_data(
    engine: Engine, run_id: int | None
) -> tuple[BarPlotData, str, str] | None:
    """Compute percentage active stim-split bar plot data without rendering."""
    data_by_condition = _query_fov_percentage_active(
        engine, run_id, include_stim_status=True
    )
    if not data_by_condition:
        return None
    plot_data = _aggregate_percentage_data_to_condition_stats(data_by_condition)
    if not plot_data["conditions"]:
        return None
    return plot_data, "Percentage Active ROIs (Stim vs NonStim)", "%"
