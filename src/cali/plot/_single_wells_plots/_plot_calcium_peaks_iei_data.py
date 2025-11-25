from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, create_engine, select

from cali.sqlmodel._model import FOV, ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _get_traces_for_run(roi_model: ROI, run_id: int | None) -> Traces | None:
    """Get the Traces object for a specific run from the ROI's traces_history."""
    if not roi_model.traces_history:
        return None
    if run_id is None:
        return roi_model.traces_history[0] if roi_model.traces_history else None
    for trace in roi_model.traces_history:
        if trace.analysis_result_id == run_id:
            return trace
    return None


def _get_data_analysis_for_run(
    roi_model: ROI, run_id: int | None
) -> DataAnalysis | None:
    """Get the DataAnalysis object for a specific run from the ROI's data_analysis_history."""
    if not roi_model.data_analysis_history:
        return None
    if run_id is None:
        return (
            roi_model.data_analysis_history[0]
            if roi_model.data_analysis_history
            else None
        )
    # First try to find exact match
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return (
        roi_model.data_analysis_history[0] if roi_model.data_analysis_history else None
    )


def _plot_iei_data(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inter-event interval data by querying database directly.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    db_path : str | Path
        Path to the SQLite database
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    run_id : int | None
        The run ID to filter by, None for latest
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Query database for ROI data
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with Session(engine) as session:
        # Get detection_settings_id from the run if run_id is provided
        detection_settings_id: int | None = None
        if run_id is not None:
            result = session.get(CaliResult, run_id)
            if result:
                detection_settings_id = result.detection_settings

        # Build query to get ROIs for this FOV with eager loading of related data
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .options(
                selectinload(ROI.data_analysis_history),  # type: ignore
            )
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Filter by detection settings if we have a run_id
        if detection_settings_id is not None:
            stmt = stmt.where(col(ROI.detection_settings_id) == detection_settings_id)

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    engine.dispose(close=True)

    for roi in roi_models:
        data_analysis = _get_data_analysis_for_run(roi, run_id)
        if data_analysis is None:
            continue
        _plot_metrics(ax, roi, data_analysis)

    _set_graph_title_and_labels(ax)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi: ROI,
    data_analysis: DataAnalysis,
) -> None:
    """Plot inter-event intervals for a single ROI."""
    if not data_analysis.iei:
        return
    # plot mean inter-event intervals +- sem of each ROI
    mean_iei = np.mean(data_analysis.iei)
    sem_iei = mean_iei / np.sqrt(len(data_analysis.iei))
    ax.errorbar(
        [roi.label_value],
        mean_iei,
        yerr=sem_iei,
        fmt="o",
        label=f"ROI {roi.label_value}",
        capsize=5,
    )
    ax.scatter(
        [roi.label_value] * len(data_analysis.iei),
        data_analysis.iei,
        alpha=0.5,
        color="lightgray",
        s=30,
        label=f"ROI {roi.label_value}",
    )


def _set_graph_title_and_labels(
    ax: Axes,
) -> None:
    """Set axis labels based on the plotted data."""
    title = "Calcium Peaks Inter-event intervals (Sec - Mean ± SEM - Deconvolved ΔF/F)"
    x_lbl = "ROIs"
    ax.set_title(title)
    ax.set_ylabel("Inter-event intervals (Sec)")
    ax.set_xlabel(x_lbl)
    if x_lbl == "ROIs":
        ax.set_xticks([])
        ax.set_xticklabels([])


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for ROI traces, not for peaks or other elements
        if label and "ROI" in label and not label.startswith("_"):
            # Get the data point coordinates
            _x, y = sel.target

            # Create hover text with ROI and value information
            roi = cast("str", label.split(" ")[1])

            # Show IEI value in seconds
            hover_text = f"{label}\nIEI: {y:.3f} sec"

            sel.annotation.set(text=hover_text, fontsize=8, color="black")

            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
