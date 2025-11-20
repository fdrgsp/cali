from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors
import numpy as np

from cali.plot._util import (
    _get_calcium_peaks_event_synchrony,
    _get_calcium_peaks_event_synchrony_matrix,
    _get_calcium_peaks_events_from_rois,
)

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from cali.gui._graph_widgets import _SingleWellGraphWidget

cali_logger = logging.getLogger("cali_logger")


def _plot_peak_event_synchrony_data(
    widget: _SingleWellGraphWidget,
    db_path: str,
    fov_name: str,
    rois: list[int] | None = None,
) -> None:
    """Plot peak event-based synchrony analysis.

    Parameters
    ----------
    widget: _SingleWellGraphWidget
        widget to plot on
    db_path: str
        Path to the database file
    fov_name: str
        Name of the FOV
    rois: list[int] | None
        List of ROI indices to include, None for all
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    peak_trains = _get_calcium_peaks_events_from_rois(db_path, fov_name, rois)
    if peak_trains is None or len(peak_trains) < 2:
        cali_logger.warning(
            "Insufficient peak data for synchrony analysis. "
            "Ensure at least two ROIs with calcium peaks are selected."
        )
        return

    jit = _get_jit(db_path, fov_name, rois)
    if jit is None:
        cali_logger.warning(
            "No valid jitter window value found for synchrony analysis."
        )
        return

    # Convert peak trains to peak event data dict for correlation-based synchrony
    peak_event_data_dict = {
        roi_name: cast("list[float]", peak_train.astype(float).tolist())
        for roi_name, peak_train in peak_trains.items()
    }

    # Use jitter window method for calcium peaks - better suited for discrete
    # events with inherent timing uncertainty due to biology and frame rate limits
    synchrony_matrix = _get_calcium_peaks_event_synchrony_matrix(
        peak_event_data_dict, method="jitter_window", jitter_window=jit
    )

    if synchrony_matrix is None:
        cali_logger.warning(
            "Failed to calculate synchrony matrix. "
            "Ensure peak event data is valid and contains sufficient data."
        )
        return

    # Calculate global synchrony metric using peak event-specific function
    global_synchrony = _get_calcium_peaks_event_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Calcium Peaks Events - Jitter Window Method)\n"
    )

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Peak Event Synchrony Index")

    ax.set_title(title)
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel("ROI")
    ax.set_xticklabels([])
    ax.set_xticks([])

    active_rois = list(peak_trains.keys())
    _add_hover_functionality(img, widget, active_rois, synchrony_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_jit(db_path: str, fov_name: str, rois: list[int] | None) -> int | None:
    """Get the jitter window value for synchrony from database."""
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, col, create_engine, select

    from cali.sqlmodel._model import FOV, ROI

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        stmt = select(ROI).join(FOV).where(col(FOV.name) == fov_name)
        if rois is not None:
            stmt = stmt.where(col(ROI.id).in_(rois))
        stmt = stmt.where(col(ROI.active) == True).options(  # noqa: E712
            selectinload(ROI.data_analysis),  # type: ignore
        )
        roi_results = session.exec(stmt).all()

    if not roi_results:
        cali_logger.warning("No valid ROIs found for synchrony analysis.")
        return None

    # Use the first ROI since the jitter window is the same for all ROIs
    roi = roi_results[0]
    if roi.data_analysis is None:
        return None

    return roi.data_analysis.calcium_sync_jitter_window


def _add_hover_functionality(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[str],
    synchrony_matrix: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[x], rois[y]
            sel.annotation.set(
                text=(
                    f"ROI {roi_x} ↔ ROI {roi_y}\n"
                    f"Peak Event Synchrony: {synchrony_matrix[y, x]:.3f}"
                ),
                fontsize=8,
                color="black",
            )
            if roi_x.isdigit() and roi_y.isdigit():
                widget.roiSelected.emit([roi_x, roi_y])
