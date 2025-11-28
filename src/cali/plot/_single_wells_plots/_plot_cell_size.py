from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

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


def _plot_cell_size_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot cell size data by querying database directly.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    run_id : int | None
        The run ID to filter by, None for latest
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Query database for ROI data
    with Session(engine) as session:
        # Get detection_settings_id from the run if run_id is provided
        detection_settings_id: int | None = None
        if run_id is None:
            cali_logger.warning("No run_id provided for cell size plot.")
            return

        result = session.get(CaliResult, run_id)
        if result:
            detection_settings_id = result.detection_settings

        # Build query to get ROIs for this FOV
        stmt = select(ROI).join(FOV).where(col(FOV.name) == fov_name)

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Filter by detection settings if we have a run_id
        if detection_settings_id is not None:
            stmt = stmt.where(col(ROI.detection_settings_id) == detection_settings_id)

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    units = ""

    for roi in roi_models:
        if roi.cell_size is None:
            continue
        if not units and roi.cell_size_units:
            units = roi.cell_size_units
        ax.scatter(roi.label_value, roi.cell_size, label=f"ROI {roi.label_value}")

    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel(f"Cell Size ({units})")
    ax.set_title("Cell Size per ROI")

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


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

            # Get the units from the y-axis label
            y_label = ax.get_ylabel()
            # Extract units from the y-axis label (e.g., "Cell Size (μm²)" -> "μm²")
            if "(" in y_label and ")" in y_label:
                units = y_label.split("(")[1].split(")")[0]
                hover_text = f"{label}\nSize: {y:.3f} {units}"
            else:
                hover_text = f"{label}\nSize: {y:.3f}"

            sel.annotation.set(text=hover_text, fontsize=8, color="black")

            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
