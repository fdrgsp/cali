from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, create_engine, select

from cali.sqlmodel._model import FOV, ROI

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _plot_cell_size_data(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    rois: list[int] | None = None,
) -> None:
    """Plot cell size data by querying database directly.

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

    units = ""

    for roi in roi_models:
        if roi.data_analysis is None or roi.data_analysis.cell_size is None:
            continue
        if not units and roi.data_analysis.cell_size_units:
            units = roi.data_analysis.cell_size_units
        ax.scatter(
            roi.label_value, roi.data_analysis.cell_size, label=f"ROI {roi.label_value}"
        )

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
