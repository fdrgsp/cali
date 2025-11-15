from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, create_engine, select

from cali.sqlmodel._model import FOV, ROI

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _plot_iei_data(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    rois: list[int] | None = None,
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
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Query database for ROI data
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with Session(engine) as session:
        # Build query to get ROIs for this FOV with eager loading of related data
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .options(
                selectinload(ROI.data_analysis),  # type: ignore
            )
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    engine.dispose(close=True)

    for roi in roi_models:
        if roi.data_analysis is None:
            continue
        _plot_metrics(ax, roi)

    _set_graph_title_and_labels(ax)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi: ROI,
) -> None:
    """Plot inter-event intervals for a single ROI."""
    if roi.data_analysis is None or not roi.data_analysis.iei:
        return
    # plot mean inter-event intervals +- sem of each ROI
    mean_iei = np.mean(roi.data_analysis.iei)
    sem_iei = mean_iei / np.sqrt(len(roi.data_analysis.iei))
    ax.errorbar(
        [roi.label_value],
        mean_iei,
        yerr=sem_iei,
        fmt="o",
        label=f"ROI {roi.label_value}",
        capsize=5,
    )
    ax.scatter(
        [roi.label_value] * len(roi.data_analysis.iei),
        roi.data_analysis.iei,
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
