from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

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


def _plot_iei_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inter-event interval data by querying the database.

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
        The run ID to filter by, None for latest (currently required)
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    if run_id is None:
        cali_logger.warning("No run_id provided for IEI plot.")
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

    # Query database for ROI + DataAnalysis
    with Session(engine) as session:
        stmt = (
            select(ROI, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data: list[tuple[ROI, DataAnalysis]] = session.exec(stmt).all()

    if not roi_data:
        ax.text(
            0.5,
            0.5,
            "No ROI analysis data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    for roi, data_analysis in roi_data:
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

    iei = np.asarray(data_analysis.iei, dtype=float)

    # Mean IEI
    mean_iei = float(np.mean(iei))

    # SEM = std / sqrt(N), not mean / sqrt(N)
    if iei.size > 1:
        std_iei = float(np.std(iei, ddof=1))
        sem_iei = std_iei / np.sqrt(iei.size)
    else:
        sem_iei = 0.0

    ax.errorbar(
        [float(roi.label_value)],
        [mean_iei],
        yerr=[sem_iei],
        fmt="o",
        label=f"ROI {roi.label_value}",
        capsize=5,
        picker=5,  # Enable picking on errorbar
    )

    ax.scatter(
        [float(roi.label_value)] * iei.size,
        iei,
        alpha=0.5,
        color="lightgray",
        s=30,
        label=f"ROI {roi.label_value}",  # Add label so scatter is pickable
        picker=True,  # Enable picking on scatter
    )


def _set_graph_title_and_labels(ax: Axes) -> None:
    """Set axis labels based on the plotted data."""
    title = "Calcium Peaks Inter-Event Intervals (s, Mean ± SEM - Deconvolved ΔF/F)"
    ax.set_title(title)
    ax.set_ylabel("Inter-Event Interval (s)")
    ax.set_xlabel("ROIs")
    ax.set_xticks([])
    ax.set_xticklabels([])


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click(ax, widget, picker_tolerance=5)
