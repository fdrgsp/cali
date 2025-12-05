"""Utility functions for multi-well bar plots.

Common functions used across multiple plot types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pyqtgraph as pg
from pyqtgraph import BarGraphItem
from sqlmodel import Session, col, select

from cali._constants import EVK_NON_STIM, EVK_STIM, EVOKED
from cali.sqlmodel import FOV, ROI, AnalysisSettings, DataAnalysis, Well

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


def _get_default_color(condition: str) -> str:
    """Get the default color for a condition based on its name.

    Parameters
    ----------
    condition : str
        Condition name (e.g., "c1_g1_evk_stim")

    Returns
    -------
    str
        Color name: "green" for evk_stim, "magenta" for evk_non_stim, "gray" otherwise
    """
    if condition.endswith(EVK_STIM):
        return "green"
    elif condition.endswith(EVK_NON_STIM):
        return "magenta"
    else:
        return "gray"


def _get_default_conditions(
    conditions: list[str],
) -> dict[str, dict[str, bool | str]]:
    """Create default conditions dict with colors based on condition names.

    Parameters
    ----------
    conditions : list[str]
        List of condition names

    Returns
    -------
    dict[str, dict[str, bool | str]]
        Dictionary mapping condition name to dict with 'visible' and 'color' keys
    """
    return {
        cond: {"visible": True, "color": _get_default_color(cond)}
        for cond in conditions
    }


class BarPlotData(TypedDict):
    """Type definition for bar plot data."""

    conditions: list[str]
    means: list[float]
    sems: list[float]
    fov_values_list: list[np.ndarray]


def _get_condition_label(
    well: Well, roi: ROI | None = None, experiment_type: str | None = None
) -> str:
    """Get a human-readable label for a well's conditions.

    Parameters
    ----------
    well : Well
        Well object with conditions
    roi : ROI | None
        Optional ROI to include stimulation status for evoked experiments
    experiment_type : str | None
        Type of experiment (e.g., "Evoked Activity", "Spontaneous Activity").
        If provided and equals EVOKED, stimulation status will be appended
        for ROIs with stimulated attribute.

    Returns
    -------
    str
        Formatted condition label (e.g., "c1_g1" for two conditions,
        or "c1_g1_evk_stim" for evoked experiments with stimulated ROIs)
    """
    if not well.conditions:
        base_label = f"Well_{well.name}"
    else:
        # Define priority order for common condition types
        # Lower priority number = appears first in label
        condition_type_priority = {
            "control": 0,
            "treatment": 1,
            "genotype": 2,
            "drug": 3,
            "timepoint": 4,
            "other": 999,
        }

        # Sort conditions by priority (then alphabetically for same priority)
        sorted_conditions = sorted(
            well.conditions,
            key=lambda c: (
                condition_type_priority.get(c.condition_type.lower(), 500),
                c.condition_type.lower(),
            ),
        )
        # Join condition names with underscore
        base_label = "_".join(c.name for c in sorted_conditions)

    # Append stimulation status ONLY for evoked experiments
    if experiment_type == EVOKED and roi is not None and roi.stimulated is not None:
        stim_suffix = EVK_STIM if roi.stimulated else EVK_NON_STIM
        return f"{base_label}_{stim_suffix}"

    return base_label


def _query_roi_parameter_by_condition(
    engine: Engine,
    parameter: str,
    run_id: int | None = None,
    include_stim_status: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """Query ROI-level parameters grouped by condition and FOV.

    Parameters
    ----------
    engine : Engine
        Database engine
    parameter : str
        Parameter name from DataAnalysis (e.g., 'dec_dff_frequency')
    run_id : int | None
        Filter by specific analysis run
    include_stim_status : bool
        If True, include stimulation status in condition labels for evoked experiments.
        Default is False (general plots don't split by stim status).

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dict: {condition_label: {fov_name: [values]}}
    """
    with Session(engine) as session:
        # Get experiment type if run_id is provided and stim status is needed
        experiment_type = None
        if include_stim_status and run_id is not None:
            from cali.sqlmodel import CaliResult

            stmt_exp_type = (
                select(AnalysisSettings.experiment_type)
                .join(
                    CaliResult, CaliResult.analysis_settings_id == AnalysisSettings.id
                )
                .where(CaliResult.id == run_id)
            )
            experiment_type = session.exec(stmt_exp_type).first()

        # Build query - start from DataAnalysis and join backwards
        stmt = (
            select(DataAnalysis, ROI, FOV, Well)
            .select_from(DataAnalysis)
            .join(ROI, DataAnalysis.roi_id == ROI.id)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .where(col(ROI.active) == True)  # noqa: E712
        )

        if run_id is not None:
            stmt = stmt.where(col(DataAnalysis.analysis_result_id) == run_id)

        results = session.exec(stmt).all()

        # Group by condition and FOV
        data: dict[str, dict[str, list[float]]] = {}
        for analysis, roi, fov, well in results:
            # Get value for this ROI
            value = getattr(analysis, parameter, None)
            if value is None:
                continue

            # Get condition label (only include stim status if requested)
            if include_stim_status:
                cond_label = _get_condition_label(well, roi, experiment_type)
            else:
                cond_label = _get_condition_label(well)

            # Initialize nested structure if needed
            if cond_label not in data:
                data[cond_label] = {}
            if fov.name not in data[cond_label]:
                data[cond_label][fov.name] = []

            data[cond_label][fov.name].append(value)

    return data


def _query_roi_attribute_by_condition(
    engine: Engine,
    attribute: str,
    run_id: int | None = None,
    include_stim_status: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """Query ROI-table attributes grouped by condition and FOV.

    This is for attributes stored on the ROI table itself (e.g., cell_size),
    not on DataAnalysis.

    Parameters
    ----------
    engine : Engine
        Database engine
    attribute : str
        Attribute name from ROI (e.g., 'cell_size')
    run_id : int | None
        Filter by specific analysis run (to get ROIs from that run)
    include_stim_status : bool
        If True, include stimulation status in condition labels for evoked experiments.
        Default is False (general plots don't split by stim status).

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dict: {condition_label: {fov_name: [values]}}
    """
    with Session(engine) as session:
        # Get experiment type if run_id is provided and stim status is needed
        experiment_type = None
        if include_stim_status and run_id is not None:
            from cali.sqlmodel import CaliResult

            stmt_exp_type = (
                select(AnalysisSettings.experiment_type)
                .join(
                    CaliResult, CaliResult.analysis_settings_id == AnalysisSettings.id
                )
                .where(CaliResult.id == run_id)
            )
            experiment_type = session.exec(stmt_exp_type).first()

        # Build query - start from ROI and join to FOV and Well
        stmt = (
            select(ROI, FOV, Well)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .where(col(ROI.active) == True)  # noqa: E712
        )

        # If run_id specified, filter by ROIs that have DataAnalysis for that run
        if run_id is not None:
            stmt = stmt.join(DataAnalysis, DataAnalysis.roi_id == ROI.id).where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        results = session.exec(stmt).all()

        # Group by condition and FOV
        data: dict[str, dict[str, list[float]]] = {}
        for roi, fov, well in results:
            # Get value for this ROI
            value = getattr(roi, attribute, None)
            if value is None:
                continue

            # Get condition label (only include stim status if requested)
            if include_stim_status:
                cond_label = _get_condition_label(well, roi, experiment_type)
            else:
                cond_label = _get_condition_label(well)

            # Initialize nested structure if needed
            if cond_label not in data:
                data[cond_label] = {}
            if fov.name not in data[cond_label]:
                data[cond_label][fov.name] = []

            data[cond_label][fov.name].append(value)

    return data


def _compute_weighted_mean_and_pooled_sem(
    fov_means: np.ndarray,
    fov_sems: np.ndarray,
    fov_ns: np.ndarray,
) -> tuple[float, float]:
    """Compute weighted mean and pooled SEM from FOV-level statistics.

    Parameters
    ----------
    fov_means : np.ndarray
        Array of means per FOV
    fov_sems : np.ndarray
        Array of SEMs per FOV
    fov_ns : np.ndarray
        Array of sample sizes per FOV

    Returns
    -------
    tuple[float, float]
        (weighted_mean, pooled_sem)
    """
    total_n = fov_ns.sum()

    if total_n <= 1:
        weighted_mean = float(fov_means.mean()) if len(fov_means) > 0 else 0.0
        pooled_sem = 0.0
    else:
        # Weighted mean
        weighted_mean = float(np.sum(fov_means * fov_ns) / total_n)

        # Pooled SEM: sqrt(sum(SEM^2 * N) / total_N)
        pooled_sem = float(np.sqrt(np.sum((fov_sems**2) * fov_ns) / total_n))

    return weighted_mean, pooled_sem


def _compute_binomial_sem(
    fov_percentages: np.ndarray,
    fov_ns: np.ndarray,
) -> tuple[float, float]:
    """Compute weighted mean and binomial SEM for percentage data.

    Parameters
    ----------
    fov_percentages : np.ndarray
        Array of percentages per FOV (0-100 scale)
    fov_ns : np.ndarray
        Array of sample sizes per FOV

    Returns
    -------
    tuple[float, float]
        (weighted_mean_percentage, binomial_sem_percentage)
    """
    total_n = fov_ns.sum()

    if total_n <= 1:
        weighted_mean = (
            float(fov_percentages.mean()) if len(fov_percentages) > 0 else 0.0
        )
        binomial_sem = 0.0
    else:
        # Convert percentages to proportions for calculation
        fov_proportions = fov_percentages / 100.0

        # Weighted mean proportion
        weighted_p = float(np.sum(fov_proportions * fov_ns) / total_n)

        # Binomial SEM: sqrt(p(1-p)/n)
        binomial_sem = float(np.sqrt(weighted_p * (1 - weighted_p) / total_n) * 100)
        weighted_mean = weighted_p * 100

    return weighted_mean, binomial_sem


def _aggregate_fov_data_to_condition_stats(
    data_by_condition: dict[str, dict[str, list[float]]],
) -> BarPlotData:
    """Aggregate FOV-level data to condition-level statistics.

    Computes mean and SEM for each FOV, then computes weighted mean
    and pooled SEM across FOVs within each condition.

    Parameters
    ----------
    data_by_condition : dict[str, dict[str, list[float]]]
        Nested dict: {condition: {fov: [roi_values]}}

    Returns
    -------
    BarPlotData
        Aggregated data ready for plotting
    """
    conditions = []
    means = []
    sems = []
    fov_values_list = []

    for cond_label, fov_dict in data_by_condition.items():
        if not fov_dict:
            continue

        # Compute mean and SEM for each FOV
        fov_means_list = []
        fov_sems_list = []
        fov_ns_list = []

        for _fov_name, roi_values in fov_dict.items():
            if not roi_values:
                continue

            # Flatten if values are lists (e.g., peaks_amplitudes_dec_dff, iei)
            # Some parameters return lists per ROI, we need to flatten them
            flat_values = []
            for val in roi_values:
                if isinstance(val, (list, np.ndarray)):
                    flat_values.extend(val)
                else:
                    flat_values.append(val)

            if not flat_values:
                continue

            values_arr = np.array(flat_values)
            n = len(values_arr)
            fov_mean = float(values_arr.mean())
            fov_sem = float(values_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

            fov_means_list.append(fov_mean)
            fov_sems_list.append(fov_sem)
            fov_ns_list.append(n)

        if not fov_means_list:
            continue

        fov_means = np.array(fov_means_list)
        fov_sems = np.array(fov_sems_list)
        fov_ns = np.array(fov_ns_list)

        # Compute weighted mean and pooled SEM
        weighted_mean, pooled_sem = _compute_weighted_mean_and_pooled_sem(
            fov_means, fov_sems, fov_ns
        )

        conditions.append(cond_label)
        means.append(weighted_mean)
        sems.append(pooled_sem)
        fov_values_list.append(fov_means)

    return BarPlotData(
        conditions=conditions,
        means=means,
        sems=sems,
        fov_values_list=fov_values_list,
    )


def _aggregate_percentage_data_to_condition_stats(
    data_by_condition: dict[str, dict[str, tuple[float, int]]],
) -> BarPlotData:
    """Aggregate FOV-level percentage data to condition-level statistics.

    Uses binomial statistics for percentage data.

    Parameters
    ----------
    data_by_condition : dict[str, dict[str, tuple[float, int]]]
        Nested dict: {condition: {fov: (percentage, n)}}

    Returns
    -------
    BarPlotData
        Aggregated data ready for plotting
    """
    conditions = []
    means = []
    sems = []
    fov_values_list = []

    for cond_label, fov_dict in data_by_condition.items():
        if not fov_dict:
            continue

        fov_percentages_list = []
        fov_ns_list = []

        for _fov_name, (percentage, n) in fov_dict.items():
            fov_percentages_list.append(percentage)
            fov_ns_list.append(n)

        if not fov_percentages_list:
            continue

        fov_percentages = np.array(fov_percentages_list)
        fov_ns = np.array(fov_ns_list)

        # Compute weighted mean and binomial SEM
        weighted_mean, binomial_sem = _compute_binomial_sem(fov_percentages, fov_ns)

        conditions.append(cond_label)
        means.append(weighted_mean)
        sems.append(binomial_sem)
        fov_values_list.append(fov_percentages)

    return BarPlotData(
        conditions=conditions,
        means=means,
        sems=sems,
        fov_values_list=fov_values_list,
    )


def _create_pyqtgraph_bar_plot(
    widget: _MultilWellGraphWidget,
    data: BarPlotData,
    parameter: str,
    units: str = "",
    title_suffix: str = "",
    bar_label: str = "Weighted Mean ± Pooled SEM",
) -> None:
    """Create a bar plot with pyqtgraph.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        Widget to plot into
    data : BarPlotData
        Plot data
    parameter : str
        Parameter name for Y-axis label
    units : str
        Units for Y-axis label
    title_suffix : str
        Additional text to append to title
    bar_label : str
        Label for the bar in the legend
    """
    # Filter based on condition toggles and respect user-defined order
    cond_list: dict[str, dict[str, bool | str]] = widget.conditions
    if not cond_list or len(cond_list) != len(data["conditions"]):
        # Initialize all conditions as enabled with default colors
        cond_list = _get_default_conditions(data["conditions"])
        widget.conditions = cond_list

    # Create a mapping from condition name to data
    data_map = {
        cond: (mean, sem, fov_vals)
        for cond, mean, sem, fov_vals in zip(
            data["conditions"],
            data["means"],
            data["sems"],
            data["fov_values_list"],
        )
    }

    # Build filtered data in the order defined by cond_list (preserves user order)
    filtered_data = [
        (cond, *data_map[cond])
        for cond in cond_list.keys()
        if cond_list[cond]["visible"] and cond in data_map
    ]

    if not filtered_data:
        widget.clear_plot()
        return

    filtered_conditions, filtered_means, filtered_sems, filtered_fov_values = map(
        list, zip(*filtered_data)
    )

    # Get colors for filtered conditions
    filtered_colors = [cond_list[cond]["color"] for cond in filtered_conditions]

    plot_item = widget.plot_item

    # X positions for bars
    x = np.arange(len(filtered_conditions))

    # Create bar graph with individual colors
    bar_graph = BarGraphItem(
        x=x,
        height=filtered_means,
        width=0.6,
        brushes=[pg.mkBrush(color) for color in filtered_colors],
    )
    plot_item.addItem(bar_graph)

    # Add error bars
    error_bars = pg.ErrorBarItem(
        x=x,
        y=filtered_means,
        height=np.array(filtered_sems),
        beam=0.2,
        pen={"color": "w", "width": 2},
    )
    plot_item.addItem(error_bars)

    # Add scatter points for individual FOV values
    for idx, fov_vals in enumerate(filtered_fov_values):
        # Add some jitter to x positions for visibility
        x_positions = np.random.normal(idx, 0.05, size=len(fov_vals))
        scatter = pg.ScatterPlotItem(
            x=x_positions,
            y=fov_vals,
            size=6,
            pen=pg.mkPen("w", width=1),
            brush=pg.mkBrush("w"),
        )
        plot_item.addItem(scatter)

    # Set up axes with abbreviated labels if needed
    bottom_axis = plot_item.getAxis("bottom")

    # Create abbreviated condition labels to prevent overlap
    # Use newlines to wrap long labels instead of rotation
    abbreviated_conditions = []
    for cond in filtered_conditions:
        # Split on underscores and create multi-line labels
        parts = cond.split("_")
        if len(parts) > 2:
            # Create label with line breaks for readability
            abbreviated = "_".join(parts[:2]) + "\n" + "_".join(parts[2:])
        else:
            abbreviated = cond
        abbreviated_conditions.append(abbreviated)

    bottom_axis.setTicks(
        [[(i, label) for i, label in enumerate(abbreviated_conditions)]]
    )
    # Increase tick text height to accommodate multi-line labels
    # Also increase bottom spacing to prevent clipping at canvas edge
    bottom_axis.setStyle(tickTextHeight=65)
    bottom_axis.setHeight(80)  # Reserve more space for the bottom axis

    units_text = f" ({units})" if units else ""
    plot_item.setLabel("left", f"{parameter}{units_text}")

    # Set title
    title = f"{parameter} per Condition{title_suffix}"
    plot_item.setTitle(title)

    # Add grid
    plot_item.showGrid(x=False, y=True, alpha=0.3)


def plot_parameter_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    parameter: str = "",
    units: str = "",
    title_suffix: str = "",
) -> None:
    """Plot a bar plot for a given parameter across conditions.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        Widget to plot into
    text : str
        Plot name (for title)
    engine : Engine
        Database engine
    run_id : int | None
        Filter by analysis run
    parameter : str
        DataAnalysis attribute name (e.g., 'dec_dff_frequency')
    units : str
        Units for Y-axis label
    title_suffix : str
        Suffix to append to plot title (e.g., "(Median)")
    """
    if not parameter:
        widget.clear_plot()
        return

    # Query data grouped by condition
    data_by_condition = _query_roi_parameter_by_condition(engine, parameter, run_id)

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
        units=units,
        title_suffix=title_suffix,
        bar_label="Weighted Mean ± Pooled SEM",
    )
