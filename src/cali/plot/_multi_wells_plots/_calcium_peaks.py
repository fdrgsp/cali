"""Calcium peaks related bar plots for multi-well analysis.

This module provides bar plot visualizations for calcium peak metrics:
- Amplitude
- Frequency
- Inter-event interval (IEI)
- Synchrony
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, col, select

from cali.sqlmodel import FOV, ROI, DataAnalysis, Traces, Well

from ._util import (
    _aggregate_fov_data_to_condition_stats,
    _create_pyqtgraph_bar_plot,
    _get_condition_label,
    plot_parameter_bar_plot,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


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
