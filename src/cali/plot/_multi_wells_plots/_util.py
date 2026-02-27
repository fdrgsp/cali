"""Utility functions for multi-well bar plots.

Common functions used across multiple plot types.
"""

from __future__ import annotations

import colorsys
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


# Palette of visually distinct colors used when no specific color is assigned.
_CONDITION_PALETTE = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

_GOLDEN_RATIO = 0.618033988749895


def _make_n_colors(n: int) -> list[str]:
    """Return n visually distinct hex color strings.

    Uses the qualitative palette for the first entries; generates additional
    evenly-spaced HSV colors (golden-ratio hue steps) beyond that so that no
    two conditions ever share the same color regardless of how many there are.
    """
    if n <= 0:
        return []
    if n <= len(_CONDITION_PALETTE):
        return list(_CONDITION_PALETTE[:n])

    colors = list(_CONDITION_PALETTE)
    for i in range(n - len(_CONDITION_PALETTE)):
        h = (i * _GOLDEN_RATIO) % 1.0
        s = 0.75 if i % 2 == 0 else 0.55
        v = 0.85 if i % 3 != 2 else 0.65
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def _get_default_conditions(
    conditions: list[str],
    multicolor: bool = False,
    override_color: str | None = None,
) -> dict[str, dict[str, bool | str]]:
    """Create default conditions dict with colors based on condition names.

    EVK_STIM conditions always get green, EVK_NON_STIM always get magenta.

    For all other conditions:
    - When ``override_color`` is set, all conditions use that fixed color (e.g.
      ``"green"`` for the stim-only bar plot, ``"magenta"`` for non-stim).
    - When ``multicolor=False`` (default, used by bar plots): every
      non-EVK condition gets "gray".  This keeps bar plots visually neutral
      so that only the stimulation-split variants use colour.
    - When ``multicolor=True`` (used by scatter / PCA plots): each non-EVK
      condition is assigned a distinct colour from the palette so that
      different conditions are visually distinguishable in the scatter space.

    Parameters
    ----------
    conditions : list[str]
        List of condition names
    multicolor : bool
        When True, assign distinct palette colours to non-EVK conditions.
        When False (default), assign "gray" to all non-EVK conditions.
    override_color : str | None
        When set, every condition receives this color regardless of its name.
        Takes precedence over both ``multicolor`` and the EVK special colors.

    Returns
    -------
    dict[str, dict[str, bool | str]]
        Dictionary mapping condition name to dict with 'visible' and 'color' keys
    """
    result: dict[str, dict[str, bool | str]] = {}

    if override_color is not None:
        for cond in conditions:
            result[cond] = {"visible": True, "color": override_color}
        return result

    if multicolor:
        # Pre-compute enough distinct colors for all non-EVK conditions so
        # that no two conditions ever share a color, even beyond 10.
        non_evk_count = sum(1 for c in conditions if _get_default_color(c) == "gray")
        extra_colors = _make_n_colors(non_evk_count)
        palette_idx = 0
        for cond in conditions:
            color = _get_default_color(cond)
            if color == "gray":
                color = extra_colors[palette_idx]
                palette_idx += 1
            result[cond] = {"visible": True, "color": color}
    else:
        # Bar-plot mode: non-EVK conditions are always gray; only
        # stim/non-stim labels carry the green/magenta signal.
        for cond in conditions:
            color = _get_default_color(cond)
            result[cond] = {"visible": True, "color": color}

    return result


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


def _get_experiment_type(session: Session, run_id: int) -> str | None:
    """Look up experiment_type for a given CaliResult run_id.

    Parameters
    ----------
    session : Session
        Active database session.
    run_id : int
        CaliResult id.

    Returns
    -------
    str | None
        The experiment type (e.g., "Spontaneous Activity", "Evoked Activity"),
        or None if not found.
    """
    from cali.sqlmodel import CaliResult

    stmt = (
        select(AnalysisSettings.experiment_type)
        .join(CaliResult, CaliResult.analysis_settings_id == AnalysisSettings.id)
        .where(CaliResult.id == run_id)
    )
    return session.exec(stmt).first()


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
        Parameter name from DataAnalysis (e.g., 'den_dff_frequency')
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
            experiment_type = _get_experiment_type(session, run_id)

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
    """Query ROI attributes grouped by condition and FOV.

    Parameters
    ----------
    engine : Engine
        Database engine
    attribute : str
        Attribute name from ROI or DataAnalysis (e.g., 'cell_size')
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
            experiment_type = _get_experiment_type(session, run_id)

        # Build query from ROI table
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
            # Get value from ROI
            value = getattr(roi, attribute, None)
            if value is None:
                continue

            # Get condition label
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


def _compute_condition_mean_and_sem(
    fov_means: np.ndarray,
) -> tuple[float, float]:
    """Compute condition mean and SEM treating each FOV as an independent replicate.

    This is the standard approach in calcium imaging: each FOV (field of view)
    is treated as a single independent observation, regardless of how many ROIs
    it contains.  The condition mean is the unweighted mean of FOV means, and
    the SEM is the standard error across FOV means.

    Parameters
    ----------
    fov_means : np.ndarray
        Array of means per FOV (one value per FOV).

    Returns
    -------
    tuple[float, float]
        (condition_mean, condition_sem)
    """
    n_fovs = len(fov_means)
    if n_fovs == 0:
        return 0.0, 0.0
    if n_fovs == 1:
        return float(fov_means[0]), 0.0

    condition_mean = float(np.mean(fov_means))
    condition_sem = float(np.std(fov_means, ddof=1) / np.sqrt(n_fovs))
    return condition_mean, condition_sem



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

        # Compute mean for each FOV (one value per FOV)
        fov_means_list = []

        for _fov_name, roi_values in fov_dict.items():
            if not roi_values:
                continue

            # Flatten if values are lists (e.g., peaks_amplitudes_den_dff, iei)
            # Some parameters return lists per ROI, we need to flatten them
            flat_values = []
            for val in roi_values:
                if isinstance(val, (list, np.ndarray)):
                    flat_values.extend(val)
                else:
                    flat_values.append(val)

            if not flat_values:
                continue

            fov_means_list.append(float(np.mean(flat_values)))

        if not fov_means_list:
            continue

        fov_means = np.array(fov_means_list)

        # Compute condition mean and SEM treating each FOV as a replicate
        condition_mean, condition_sem = _compute_condition_mean_and_sem(fov_means)

        conditions.append(cond_label)
        means.append(condition_mean)
        sems.append(condition_sem)
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

        for _fov_name, (percentage, _n) in fov_dict.items():
            fov_percentages_list.append(percentage)

        if not fov_percentages_list:
            continue

        fov_percentages = np.array(fov_percentages_list)

        # Compute mean and SEM treating each FOV percentage as a replicate
        mean_pct, sem_pct = _compute_condition_mean_and_sem(fov_percentages)

        conditions.append(cond_label)
        means.append(mean_pct)
        sems.append(sem_pct)
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
    bar_label: str = "Mean ± SEM (per FOV)",
    override_color: str | None = None,
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
    override_color : str | None
        When set, all bars are painted this color regardless of condition names.
        Use ``"green"`` for stim-only plots and ``"magenta"`` for non-stim plots.
    """
    # Filter based on condition toggles and respect user-defined order
    cond_list: dict[str, dict[str, bool | str]] = widget.conditions
    if override_color is not None:
        # Always re-initialize with the fixed override color so the bars
        # are always the right color regardless of cached state.
        cond_list = _get_default_conditions(
            data["conditions"], override_color=override_color
        )
        widget.conditions = cond_list
    elif not cond_list or len(cond_list) != len(data["conditions"]):
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
        pen={"color": "k", "width": 2},
    )
    plot_item.addItem(error_bars)

    # Add scatter points for individual FOV values
    rng = np.random.default_rng(42)
    for idx, fov_vals in enumerate(filtered_fov_values):
        # Add some jitter to x positions for visibility
        x_positions = rng.normal(idx, 0.05, size=len(fov_vals))
        scatter = pg.ScatterPlotItem(
            x=x_positions,
            y=fov_vals,
            size=6,
            pen=pg.mkPen("k", width=1),
            brush=pg.mkBrush("k"),
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
    title_error = f" — {bar_label}" if bar_label else ""
    title = f"{parameter} per Condition{title_suffix}{title_error}"
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
    include_stim_status: bool = False,
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
        DataAnalysis attribute name (e.g., 'den_dff_frequency')
    units : str
        Units for Y-axis label
    title_suffix : str
        Suffix to append to plot title (e.g., "(Median)")
    include_stim_status : bool
        If True, condition labels include stim/non-stim split (evoked plots only).
    """
    if not parameter:
        widget.clear_plot()
        return

    # Query data grouped by condition
    data_by_condition = _query_roi_parameter_by_condition(
        engine, parameter, run_id, include_stim_status=include_stim_status
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
        units=units,
        title_suffix=title_suffix,
        bar_label="Mean ± SEM (per FOV)",
    )
