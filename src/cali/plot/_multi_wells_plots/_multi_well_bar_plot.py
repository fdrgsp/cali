"""Multi-well bar plots using database queries.

This module provides bar plot visualization across multiple wells/conditions,
querying data directly from the database instead of relying on CSV files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pyqtgraph as pg
from pyqtgraph import BarGraphItem
from sqlmodel import Session, col, select

from cali._constants import EVK_NON_STIM, EVK_STIM, EVOKED
from cali.sqlmodel import FOV, ROI, AnalysisSettings, DataAnalysis, Traces, Well

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
        # Sort conditions by condition_type to ensure consistent naming order
        # (e.g., genotype always before treatment)
        sorted_conditions = sorted(well.conditions, key=lambda c: c.condition_type)
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

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dict: {condition_label: {fov_name: [values]}}
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

            # Get condition label (including stimulation status for evoked exps)
            cond_label = _get_condition_label(well, roi, experiment_type)

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

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dict: {condition_label: {fov_name: [values]}}
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

            # Get condition label (including stimulation status for evoked exps)
            cond_label = _get_condition_label(well, roi, experiment_type)

            # Initialize nested structure if needed
            if cond_label not in data:
                data[cond_label] = {}
            if fov.name not in data[cond_label]:
                data[cond_label][fov.name] = []

            data[cond_label][fov.name].append(value)

    return data


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

    # Build filtered data in the order defined by cond_list (which preserves user order)
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

    # Set up axes
    bottom_axis = plot_item.getAxis("bottom")
    bottom_axis.setTicks([[(i, cond) for i, cond in enumerate(filtered_conditions)]])

    units_text = f" ({units})" if units else ""
    plot_item.setLabel("left", f"{parameter}{units_text}")

    # Set title
    title = f"{parameter} per Condition{title_suffix}"
    plot_item.setTitle(title)

    # Add grid
    plot_item.showGrid(x=False, y=True, alpha=0.3)


def _query_spike_synchrony_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, float]]:
    """Query spike synchrony per FOV, grouped by condition.

    Calculates synchrony on-the-fly from inferred spikes.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {condition: {fov_name: synchrony_value}}
    """
    from cali.plot._util import _get_spike_synchrony, _get_spike_synchrony_matrix

    with Session(engine) as session:
        # Query all ROIs with traces and analysis grouped by FOV
        stmt = (
            select(ROI, FOV, Well, Traces, DataAnalysis)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Traces, ROI.id == Traces.roi_id)
            .join(DataAnalysis, ROI.id == DataAnalysis.roi_id)
        )

        if run_id is not None:
            stmt = stmt.where(col(Traces.analysis_result_id) == run_id).where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        # Only active ROIs
        stmt = stmt.where(col(ROI.active) == True)  # noqa: E712
        stmt = stmt.order_by(col(FOV.id), col(ROI.label_value))

        results = session.exec(stmt).all()

        # Group by condition and FOV
        data: dict[str, dict[str, float]] = {}

        # Organize by FOV first
        fov_data: dict[tuple[str, str], list[tuple[ROI, Traces, DataAnalysis]]] = {}
        for roi, fov, well, traces, analysis in results:
            cond_label = _get_condition_label(well, fov.name)
            key = (cond_label, fov.name)
            fov_data.setdefault(key, []).append((roi, traces, analysis))

        # Calculate synchrony for each FOV
        for (cond_label, fov_name), roi_list in fov_data.items():
            if len(roi_list) < 2:
                # Need at least 2 ROIs for synchrony
                continue

            # Build spike data dict (thresholded spikes)
            spike_data: dict[str, list[float]] = {}
            for roi, traces, analysis in roi_list:
                if (
                    not traces.inferred_spikes
                    or analysis.inferred_spikes_threshold is None
                ):
                    continue

                # Threshold the spikes
                inferred_spikes = np.array(traces.inferred_spikes)
                threshold = analysis.inferred_spikes_threshold
                spikes_thresholded = np.where(
                    inferred_spikes > threshold, inferred_spikes, 0.0
                ).tolist()

                roi_key = f"ROI_{roi.label_value}"
                spike_data[roi_key] = spikes_thresholded

            if len(spike_data) < 2:
                continue

            # Calculate synchrony matrix
            sync_matrix = _get_spike_synchrony_matrix(spike_data)

            # Calculate global synchrony (median)
            if sync_matrix is not None:
                global_sync = _get_spike_synchrony(sync_matrix)

                if global_sync is not None:
                    data.setdefault(cond_label, {})[fov_name] = global_sync

    return data


def _query_calcium_network_density_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, float]]:
    """Query calcium network density per FOV, grouped by condition.

    Calculates network density from correlation matrix on-the-fly.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {condition: {fov_name: density_percentage}}
    """
    from scipy.signal import correlate
    from scipy.stats import zscore

    from cali.plot._util import _create_connectivity_matrix

    with Session(engine) as session:
        # Query all ROIs with traces grouped by FOV
        stmt = (
            select(ROI, FOV, Well, Traces)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Traces, ROI.id == Traces.roi_id)
        )

        if run_id is not None:
            stmt = stmt.where(col(Traces.analysis_result_id) == run_id)

        # Only active ROIs
        stmt = stmt.where(col(ROI.active) == True)  # noqa: E712
        stmt = stmt.order_by(col(FOV.id), col(ROI.label_value))

        results = session.exec(stmt).all()

        # Group by condition and FOV
        data: dict[str, dict[str, float]] = {}

        # Organize by FOV first
        fov_data: dict[tuple[str, str], list[tuple[ROI, Traces]]] = {}
        for roi, fov, well, traces in results:
            cond_label = _get_condition_label(well)
            key = (cond_label, fov.name)
            fov_data.setdefault(key, []).append((roi, traces))

        # Calculate network density for each FOV
        for (cond_label, fov_name), roi_list in fov_data.items():
            if len(roi_list) < 2:
                # Need at least 2 ROIs for correlation
                continue

            # Collect traces
            traces_list: list[np.ndarray] = []
            for _roi, roi_traces in roi_list:
                if roi_traces.dec_dff is None:
                    continue

                tr = np.asarray(roi_traces.dec_dff, dtype=float)
                if tr.ndim != 1 or tr.size == 0:
                    continue

                traces_list.append(tr)

            if len(traces_list) < 2:
                continue

            # Calculate correlation matrix
            traces_array = np.vstack(traces_list)  # (n_rois, n_frames)
            dff_zero_mean = zscore(traces_array, axis=1)

            n_rois = len(traces_list)
            correlation_matrix = np.empty((n_rois, n_rois), dtype=float)

            norms = np.linalg.norm(dff_zero_mean, axis=1)
            norms[norms == 0] = np.finfo(float).eps

            np.fill_diagonal(correlation_matrix, 1.0)

            for i in range(n_rois):
                x = dff_zero_mean[i]
                for j in range(i + 1, n_rois):
                    y = dff_zero_mean[j]
                    corr = correlate(x, y, mode="full", method="fft")
                    corr /= norms[i] * norms[j]
                    max_corr = float(np.max(corr))
                    correlation_matrix[i, j] = max_corr
                    correlation_matrix[j, i] = max_corr

            # Create connectivity matrix (using default 90th percentile threshold)
            network_threshold = 90.0
            connectivity_matrix = _create_connectivity_matrix(
                correlation_matrix, network_threshold
            )

            # Calculate network density as percentage
            n_edges = np.sum(connectivity_matrix) - n_rois  # Exclude diagonal
            total_possible_edges = n_rois * (n_rois - 1)
            if total_possible_edges > 0:
                network_density = (n_edges / total_possible_edges) * 100.0
            else:
                network_density = 0.0

            data.setdefault(cond_label, {})[fov_name] = network_density

    return data


def _query_burst_metrics_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Query burst metrics per FOV, grouped by condition.

    Calculates burst count, avg duration, and avg interval on-the-fly
    from population spike activity.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Nested dict: {condition: {fov_name: {"count": ..., "avg_duration_sec": ...,
        "avg_interval_sec": ...}}}
    """
    from scipy.ndimage import gaussian_filter1d

    from cali.plot._single_wells_plots.burst._plot_inferred_spike_burst_activity import (  # noqa: E501
        _detect_population_bursts,
        _get_burst_parameters,
        _get_population_spike_data,
    )

    # Get burst detection parameters
    burst_params = _get_burst_parameters(engine, fov_name="", rois=None, run_id=run_id)
    if burst_params is None:
        return {}

    burst_threshold, min_burst_duration, smoothing_sigma = burst_params

    with Session(engine) as session:
        # Get all FOV names grouped by condition
        stmt = (
            select(FOV, Well)
            .select_from(FOV)
            .join(Well, FOV.well_id == Well.id)
            .distinct()
        )
        fov_well_results = session.exec(stmt).all()

        data: dict[str, dict[str, dict[str, float]]] = {}

        for fov, well in fov_well_results:
            cond_label = _get_condition_label(well)

            # Get population spike data for this FOV
            spike_trains, _, time_axis = _get_population_spike_data(
                engine, fov.name, rois=None, run_id=run_id
            )

            if spike_trains is None or len(spike_trains) < 2:
                continue

            # Calculate population activity
            population_activity = np.mean(spike_trains, axis=0)

            # Smooth population activity for burst detection
            smoothed_activity = gaussian_filter1d(
                population_activity, sigma=smoothing_sigma
            )

            # Detect bursts (threshold is percentage, convert to fraction)
            bursts = _detect_population_bursts(
                smoothed_activity, burst_threshold / 100, min_burst_duration
            )

            # Calculate burst statistics
            burst_count = len(bursts)

            # Calculate burst rate (bursts per minute)
            total_time_min = (time_axis[-1] - time_axis[0]) / 60.0
            burst_rate = burst_count / total_time_min if total_time_min > 0 else 0.0

            if burst_count == 0:
                # No bursts detected
                burst_metrics = {
                    "count": 0.0,
                    "avg_duration_sec": 0.0,
                    "avg_interval_sec": 0.0,
                    "rate_per_min": 0.0,
                }
            else:
                # Calculate durations and intervals
                durations = []
                intervals = []

                for i, (start, end) in enumerate(bursts):
                    # Convert indices to time
                    duration_sec = (end - start) * (time_axis[1] - time_axis[0])
                    durations.append(duration_sec)

                    # Calculate interval to next burst
                    if i < len(bursts) - 1:
                        next_start = bursts[i + 1][0]
                        interval_sec = (next_start - end) * (
                            time_axis[1] - time_axis[0]
                        )
                        intervals.append(interval_sec)

                # Calculate statistics
                avg_duration = float(np.mean(durations)) if durations else 0.0
                avg_interval = float(np.mean(intervals)) if intervals else 0.0

                burst_metrics = {
                    "count": float(burst_count),
                    "avg_duration_sec": avg_duration,
                    "avg_interval_sec": avg_interval,
                    "rate_per_min": float(burst_rate),
                }

            data.setdefault(cond_label, {})[fov.name] = burst_metrics

    return data


def _query_calcium_peaks_synchrony_by_condition(
    engine: Engine,
    run_id: int | None = None,
) -> dict[str, dict[str, float]]:
    """Query calcium peaks synchrony per FOV, grouped by condition.

    Calculates synchrony on-the-fly from peak events.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {condition: {fov_name: synchrony_value}}
    """
    from cali.plot._util import (
        _get_calcium_peaks_event_synchrony,
        _get_calcium_peaks_event_synchrony_matrix,
    )

    with Session(engine) as session:
        # Query all ROIs with traces and analysis grouped by FOV
        stmt = (
            select(ROI, FOV, Well, Traces, DataAnalysis)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Traces, ROI.id == Traces.roi_id)
            .join(DataAnalysis, ROI.id == DataAnalysis.roi_id)
        )

        if run_id is not None:
            stmt = stmt.where(col(Traces.analysis_result_id) == run_id).where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        # Only active ROIs
        stmt = stmt.where(col(ROI.active) == True)  # noqa: E712
        stmt = stmt.order_by(col(FOV.id), col(ROI.label_value))

        results = session.exec(stmt).all()

        # Group by condition and FOV
        data: dict[str, dict[str, float]] = {}

        # Organize by FOV first
        fov_data: dict[tuple[str, str], list[tuple[ROI, Traces, DataAnalysis]]] = {}
        for roi, fov, well, traces, analysis in results:
            cond_label = _get_condition_label(well, fov.name)
            key = (cond_label, fov.name)
            fov_data.setdefault(key, []).append((roi, traces, analysis))

        # Calculate synchrony for each FOV
        for (cond_label, fov_name), roi_list in fov_data.items():
            if len(roi_list) < 2:
                # Need at least 2 ROIs for synchrony
                continue

            # Determine max frames for this FOV
            max_frames = 0
            for _, traces, analysis in roi_list:
                if traces.corrected_trace is not None:
                    max_frames = max(max_frames, len(traces.corrected_trace))
                if analysis.peaks_dec_dff:
                    max_peak = max((int(p) for p in analysis.peaks_dec_dff), default=0)
                    max_frames = max(max_frames, max_peak + 1)

            if max_frames == 0:
                continue

            # Build binary peak event arrays
            peak_event_data: dict[str, np.ndarray] = {}
            for roi, _, analysis in roi_list:
                if not analysis.peaks_dec_dff:
                    continue

                # Create binary peak event train
                peak_train = np.zeros(max_frames, dtype=np.float32)
                for peak_frame in analysis.peaks_dec_dff:
                    if 0 <= int(peak_frame) < max_frames:
                        peak_train[int(peak_frame)] = 1.0

                if np.sum(peak_train) > 0:  # Only include ROIs with at least one peak
                    roi_key = f"ROI_{roi.label_value}"
                    peak_event_data[roi_key] = peak_train

            if len(peak_event_data) < 2:
                continue

            # Calculate synchrony matrix
            sync_matrix = _get_calcium_peaks_event_synchrony_matrix(peak_event_data)

            # Calculate global synchrony (median)
            if sync_matrix is not None:
                global_sync = _get_calcium_peaks_event_synchrony(sync_matrix)

                if global_sync is not None:
                    data.setdefault(cond_label, {})[fov_name] = global_sync

    return data


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


def plot_percentage_active_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot percentage of active ROIs per condition.

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
    """
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


# Specific plot functions for each parameter ==========================================


def plot_cell_size_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot cell size across conditions.

    Cell size is stored on the ROI table, not DataAnalysis, so we need
    a custom query.
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


def plot_calcium_peaks_synchrony_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium peak events global synchrony across conditions."""
    # Query synchrony data (one value per FOV)
    data_by_condition = _query_calcium_peaks_synchrony_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Convert to format expected by aggregation
    # Since we have single values per FOV, wrap in lists
    data_as_lists: dict[str, dict[str, list[float]]] = {}
    for condition, fov_dict in data_by_condition.items():
        for fov_name, sync_value in fov_dict.items():
            data_as_lists.setdefault(condition, {})[fov_name] = [sync_value]

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
        title_suffix=" (Median)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_spike_synchrony_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes global synchrony across conditions."""
    # Query synchrony data (one value per FOV)
    data_by_condition = _query_spike_synchrony_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Convert to format expected by aggregation
    # Since we have single values per FOV, wrap in lists
    data_as_lists: dict[str, dict[str, list[float]]] = {}
    for condition, fov_dict in data_by_condition.items():
        for fov_name, sync_value in fov_dict.items():
            data_as_lists.setdefault(condition, {})[fov_name] = [sync_value]

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
        title_suffix=" (Median - Thresholded Data)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_calcium_network_density_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot calcium network density across conditions.

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
    """
    # Query network density data (calculated on-the-fly)
    data_by_condition = _query_calcium_network_density_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Wrap single values in lists for aggregation
    data_for_aggregation: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        data_for_aggregation[cond] = {fov: [val] for fov, val in fov_dict.items()}

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(data_for_aggregation)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="%",
        title_suffix="",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_count_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst count across conditions.

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
    """
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract burst count from metrics dict
    count_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        count_data[cond] = {
            fov: [metrics["count"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(count_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="Count",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_avg_duration_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst average duration across conditions.

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
    """
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract avg duration from metrics dict
    duration_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        duration_data[cond] = {
            fov: [metrics["avg_duration_sec"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(duration_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="s",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_avg_interval_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst average interval across conditions.

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
    """
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract avg interval from metrics dict
    interval_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        interval_data[cond] = {
            fov: [metrics["avg_interval_sec"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(interval_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="s",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_burst_rate_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot burst rate across conditions.

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
    """
    # Query burst metrics (calculated on-the-fly)
    data_by_condition = _query_burst_metrics_by_condition(engine, run_id)

    if not data_by_condition:
        widget.clear_plot()
        return

    # Extract burst rate from metrics dict
    rate_data: dict[str, dict[str, list[float]]] = {}
    for cond, fov_dict in data_by_condition.items():
        rate_data[cond] = {
            fov: [metrics["rate_per_min"]] for fov, metrics in fov_dict.items()
        }

    # Aggregate to condition-level statistics
    plot_data = _aggregate_fov_data_to_condition_stats(rate_data)

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="bursts/min",
        title_suffix="(Inferred Spikes)",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def _query_evoked_amplitudes_by_condition(
    engine: Engine,
    stimulated: bool = True,
    run_id: int | None = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Query evoked amplitudes grouped by condition → FOV → power_pulse.

    Calculates stimulated/non-stimulated amplitudes on-the-fly from traces data.

    Parameters
    ----------
    engine : Engine
        Database engine
    stimulated : bool
        If True, get stimulated peaks; if False, get non-stimulated peaks
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    dict[str, dict[str, dict[str, list[float]]]]
        Nested dict: {condition: {fov: {power_pulse: [amplitudes]}}}
    """
    from cali.plot._util import separate_stimulated_vs_non_stimulated_peaks

    with Session(engine) as session:
        # Build query for all ROIs with their traces and analysis
        stmt = (
            select(ROI, FOV, Well, Traces, DataAnalysis)
            .select_from(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Traces, ROI.id == Traces.roi_id)
            .join(DataAnalysis, ROI.id == DataAnalysis.roi_id)
        )

        if run_id is not None:
            stmt = stmt.where(col(Traces.analysis_result_id) == run_id).where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        # Only get active ROIs
        stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

        results = session.exec(stmt).all()

        # Group by condition and FOV
        data: dict[str, dict[str, dict[str, list[float]]]] = {}

        for roi, fov, well, traces, analysis in results:
            # Check if this is an evoked experiment
            if not traces.stimulations_frames_and_powers:
                continue

            # Get stimulated/non-stimulated amplitudes
            amps_stim, amps_non_stim = separate_stimulated_vs_non_stimulated_peaks(
                dec_dff=np.array(traces.dec_dff),
                peaks_dec_dff=(
                    np.array(analysis.peaks_dec_dff)
                    if analysis.peaks_dec_dff
                    else np.array([])
                ),
                pulse_on_frames_and_powers=traces.stimulations_frames_and_powers,
                is_roi_stimulated=roi.stimulated,
                led_pulse_duration=traces.led_pulse_duration or "unknown",
                led_power_equation=None,
            )

            # Select which dict to use
            amps = amps_stim if stimulated else amps_non_stim
            if not amps:
                continue

            # Build condition label
            cond_label = _get_condition_label(well, fov.name)

            # Store amplitudes grouped by power_pulse
            for power_pulse, amplitude_list in amps.items():
                data.setdefault(cond_label, {}).setdefault(fov.name, {}).setdefault(
                    power_pulse, []
                ).extend(amplitude_list)

    return data


def _aggregate_evoked_data_to_condition_stats(
    data_by_condition: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, BarPlotData]:
    """Aggregate evoked amplitude data to condition-level statistics per power/pulse.

    Parameters
    ----------
    data_by_condition : dict[str, dict[str, dict[str, list[float]]]]
        Nested dict: {condition: {fov: {power_pulse: [amplitudes]}}}

    Returns
    -------
    dict[str, BarPlotData]
        Dict mapping power_pulse to aggregated plot data
    """
    # First, reorganize by power_pulse → condition → fov → values
    by_power_pulse: dict[str, dict[str, dict[str, list[float]]]] = {}

    for condition, fov_dict in data_by_condition.items():
        for fov, power_pulse_dict in fov_dict.items():
            for power_pulse, amplitudes in power_pulse_dict.items():
                by_power_pulse.setdefault(power_pulse, {}).setdefault(
                    condition, {}
                ).setdefault(fov, []).extend(amplitudes)

    # Now aggregate each power_pulse group
    result = {}
    for power_pulse, condition_data in by_power_pulse.items():
        result[power_pulse] = _aggregate_fov_data_to_condition_stats(condition_data)

    return result


def plot_stimulated_peaks_amplitude_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot stimulated calcium peaks amplitude across conditions.

    For evoked experiments. Creates separate plots for each LED power/pulse combination.
    """
    # Query stimulated amplitudes
    data_by_condition = _query_evoked_amplitudes_by_condition(
        engine, stimulated=True, run_id=run_id
    )

    if not data_by_condition:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # Aggregate by power/pulse
    plot_data_by_power = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    if not plot_data_by_power:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # For now, plot the first power/pulse combination
    # TODO: Add UI to select which power/pulse to display
    power_pulse = next(iter(plot_data_by_power.keys()))
    plot_data = plot_data_by_power[power_pulse]

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="ΔF/F0",
        title_suffix=f" ({power_pulse})",
        bar_label="Weighted Mean ± Pooled SEM",
    )


def plot_non_stimulated_peaks_amplitude_bar_plot(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot non-stimulated calcium peaks amplitude across conditions.

    For evoked experiments. Creates separate plots for each LED power/pulse combination.
    """
    # Query non-stimulated amplitudes
    data_by_condition = _query_evoked_amplitudes_by_condition(
        engine, stimulated=False, run_id=run_id
    )

    if not data_by_condition:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # Aggregate by power/pulse
    plot_data_by_power = _aggregate_evoked_data_to_condition_stats(data_by_condition)

    if not plot_data_by_power:
        widget.clear_plot()
        widget.plot_widget.setTitle(f"{text}<br>(No Data)")
        return

    # For now, plot the first power/pulse combination
    # TODO: Add UI to select which power/pulse to display
    power_pulse = next(iter(plot_data_by_power.keys()))
    plot_data = plot_data_by_power[power_pulse]

    if not plot_data["conditions"]:
        widget.clear_plot()
        return

    # Create the plot
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=plot_data,
        parameter=text,
        units="ΔF/F0",
        title_suffix=f" ({power_pulse})",
        bar_label="Weighted Mean ± Pooled SEM",
    )
