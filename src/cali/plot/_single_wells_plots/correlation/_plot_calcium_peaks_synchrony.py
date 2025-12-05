from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from cali.logger import cali_logger
from cali.plot._util import (
    _get_calcium_peaks_event_synchrony,
    _get_calcium_peaks_event_synchrony_matrix,
    _get_calcium_peaks_events_from_rois,
)
from cali.sqlmodel._model import ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Helpers: retrieval from ROI histories (kept for compatibility)
# -----------------------------------------------------------------------------#
def _get_traces_for_run(roi_model: ROI, run_id: int | None) -> Traces | None:
    """Get the Traces object for a specific run from the ROI's traces_history."""
    if not roi_model.traces_history:
        return None
    if run_id is None:
        return roi_model.traces_history[0]
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
        return roi_model.data_analysis_history[0]
    # First try to find exact match
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility)
    return roi_model.data_analysis_history[0]


# -----------------------------------------------------------------------------#
# Main plotting entry point (pyqtgraph version)
# -----------------------------------------------------------------------------#
def _plot_peak_event_synchrony_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
) -> None:
    """Plot peak event-based synchrony analysis (pyqtgraph heatmap).

    title_suffix : str
        Optional suffix to add to plot titles (e.g., " - Stimulated")
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend if present (we don't want it here)
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # 1) Get peak trains per ROI
    peak_trains = _get_calcium_peaks_events_from_rois(engine, fov_name, rois, run_id)
    if peak_trains is None or len(peak_trains) < 2:
        cali_logger.warning(
            "Insufficient peak data for synchrony analysis. "
            "Ensure at least two ROIs with peak events are selected."
        )
        plot.setTitle(f"Peak Event Synchrony\n(No data){title_suffix}")
        plot.setLabel("bottom", "ROI index")
        plot.setLabel("left", "ROI index")
        return

    # 2) Get jitter window from settings
    jit = _get_jit(engine, fov_name, rois, run_id)
    if jit is None:
        cali_logger.warning(
            "No valid jitter window value found for synchrony analysis."
        )
        plot.setTitle("Peak Event Synchrony\n(No jitter window)")
        plot.setLabel("bottom", "ROI index")
        plot.setLabel("left", "ROI index")
        return

    # 3) Build peak event data dict (ROI -> list[float])
    peak_event_data_dict = {
        roi_name: peak_train.astype(float).tolist()
        for roi_name, peak_train in peak_trains.items()
    }

    # 4) Compute synchrony matrix once (jitter window method)
    synchrony_matrix = _get_calcium_peaks_event_synchrony_matrix(
        peak_event_data_dict,
        method="jitter_window",
        jitter_window=jit,
    )
    if synchrony_matrix is None:
        cali_logger.warning(
            "Failed to calculate synchrony matrix. "
            "Ensure peak event data is valid and contains sufficient data."
        )
        plot.setTitle(f"Peak Event Synchrony\n(Failed to compute matrix){title_suffix}")
        plot.setLabel("bottom", "ROI index")
        plot.setLabel("left", "ROI index")
        return

    # 5) Global synchrony metric
    global_synchrony = _get_calcium_peaks_event_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    base_title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Calcium Peaks Events - Jitter Window Method){title_suffix}"
    )

    sync = np.asarray(synchrony_matrix, dtype=float)

    # ---------------- IMAGE ITEM (centered-ish, square) ---------------- #
    img = pg.ImageItem(sync)

    # viridis colormap
    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))  # fixed [0, 1]

    plot.addItem(img)

    vb = plot.getViewBox()

    # Make (0,0) top-left like imshow
    vb.invertY(True)

    # keep it square
    vb.setAspectLocked(True)  # or vb.setAspectLocked(True, ratio=1)

    plot.setTitle(base_title)
    plot.setLabel("bottom", "ROI index")
    plot.setLabel("left", "ROI index")

    # Hide axis tick labels (same behaviour as MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Synchrony")

    # Use same ROI ordering as in peak_trains.keys()
    active_roi_ids = [int(roi_id) for roi_id in peak_trains.keys()]

    # ---------------- Hover + Click interaction ---------------- #
    _attach_synchrony_heatmap_interaction(
        widget,
        plot,
        vb,
        active_roi_ids,
        sync,
        base_title=base_title,
    )


# -----------------------------------------------------------------------------#
# Settings: jitter window retrieval (unchanged from MPL version)
# -----------------------------------------------------------------------------#
def _get_jit(
    engine: Engine, fov_name: str, rois: list[int] | None, run_id: int | None = None
) -> int | None:
    """Get the jitter window value for synchrony from database."""
    from sqlmodel import Session, select

    from cali.sqlmodel._model import AnalysisSettings

    with Session(engine) as session:
        # Prefer settings from the given run
        if run_id is not None:
            result = session.get(CaliResult, run_id)
            if result and result.analysis_settings_id is not None:
                settings = session.get(AnalysisSettings, result.analysis_settings_id)
                if settings:
                    return settings.calcium_sync_jitter_window  # type: ignore[no-any-return]

        # Fallback: get settings from the first available run
        stmt = (
            select(CaliResult)
            .where(CaliResult.analysis_settings_id.is_not(None))  # type: ignore
            .limit(1)
        )
        result = session.exec(stmt).first()
        if result and result.analysis_settings_id is not None:
            settings = session.get(AnalysisSettings, result.analysis_settings_id)
            if settings:
                return settings.calcium_sync_jitter_window  # type: ignore[no-any-return]

    cali_logger.warning("No valid analysis settings found for synchrony analysis.")
    return None


# -----------------------------------------------------------------------------#
# Hover + click helper (synchrony-specific)
# -----------------------------------------------------------------------------#
def _attach_synchrony_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
    base_title: str,
) -> None:
    """
    Attach interaction to the synchrony heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with a list [roi_i, roi_j]
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # Avoid stacking multiple handlers on repeated calls
    old_hover = plot.property("sync_hover_handler")
    old_click = plot.property("sync_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    def _on_mouse_moved(pos: pg.Point) -> None:
        if not plot.sceneBoundingRect().contains(pos):
            plot.setTitle(base_title)
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = round(mouse_point.x())
        row = round(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            val = float(values[row, col])
            plot.setTitle(f"{base_title}\nROI {roi_i} vs ROI {roi_j}: {val:.3f}")
        else:
            plot.setTitle(base_title)

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = round(mouse_point.x())
        row = round(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            # Emit as list[str] to match your highlight logic
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect next time
    plot.setProperty("sync_hover_handler", _on_mouse_moved)
    plot.setProperty("sync_click_handler", _on_mouse_clicked)


def _add_colorbar_to_widget(
    widget: _SingleWellGraphWidget,
    vmin: float,
    vmax: float,
    label: str = "Synchrony",
) -> None:
    """Add a ColorBarItem to the widget layout."""
    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Create ColorBarItem
    widget.colorbar = pg.ColorBarItem(
        values=(vmin, vmax),
        colorMap=pg.colormap.get("viridis"),
        width=15,
        label=label,
        interactive=False,
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)
