from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click
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
    """Get DataAnalysis for a specific run from ROI's data_analysis_history."""
    if not roi_model.data_analysis_history:
        return None
    if run_id is None:
        return (
            roi_model.data_analysis_history[0]
            if roi_model.data_analysis_history
            else None
        )
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
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
    """Plot cell size per ROI by querying the database.

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
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    if run_id is None:
        cali_logger.warning("No run_id provided for cell size plot.")
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Query database for ROI data
    with Session(engine) as session:
        detection_settings_id: int | None = None

        result = session.get(CaliResult, run_id)
        if result:
            detection_settings_id = result.detection_settings_id

        stmt = select(ROI).join(FOV).where(col(FOV.name) == fov_name)

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        if detection_settings_id is not None:
            stmt = stmt.where(col(ROI.detection_settings_id) == detection_settings_id)

        # Only include ROIs that have cell_size data
        stmt = stmt.where(col(ROI.cell_size).is_not(None))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_models = session.exec(stmt).all()

    if not roi_models:
        ax.text(
            0.5,
            0.5,
            "No cell size data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Plot data
    units = ""
    for roi in roi_models:
        if roi.cell_size is None:
            continue
        if not units and roi.cell_size_units:
            units = roi.cell_size_units
        ax.scatter(
            float(roi.label_value),
            float(roi.cell_size),
            label=f"ROI {roi.label_value}",
            picker=True,  # Enable picking on scatter
            s=50,  # Larger size for easier clicking
        )

    # Fallback units if nothing was set
    if not units:
        units = "a.u."

    ax.set_xlabel("ROI")
    ax.set_xticks([])  # hide tick labels, keep ROI implied by hover
    ax.set_xticklabels([])
    ax.set_ylabel(f"Cell Size ({units})")
    ax.set_title("Cell Size per ROI")

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click(ax, widget, picker_tolerance=5)
