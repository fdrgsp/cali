from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from scipy.signal import correlate
from scipy.stats import zscore
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Helpers: retrieval from ROI histories
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
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    return roi_model.data_analysis_history[0]


# -----------------------------------------------------------------------------#
# Cross-correlation computation
# -----------------------------------------------------------------------------#
def _calculate_cross_correlation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for the active ROIs.

    Value = maximum normalized cross-correlation over all lags.
    ROIs are indexed by ROI.label_value (to match your other plots).
    """
    if run_id is None:
        cali_logger.warning("No run ID specified for cross-correlation plot.")
        return None, None

    with Session(engine) as session:
        stmt = (
            select(ROI, Traces)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        # IMPORTANT: use label_value for ROI subset, not ROI.id
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        roi_data: list[tuple[ROI, Traces]] = session.exec(stmt).all()

    traces: list[np.ndarray] = []
    rois_idxs: list[int] = []

    for roi, roi_traces in roi_data:
        if roi_traces is None or roi_traces.dec_dff is None or roi.label_value is None:
            continue

        tr = np.asarray(roi_traces.dec_dff, dtype=float)
        if tr.ndim != 1 or tr.size == 0:
            continue

        rois_idxs.append(int(roi.label_value))
        traces.append(tr)

    if len(rois_idxs) <= 1:
        cali_logger.warning(
            "Not enough active ROIs to calculate cross-correlation. "
            "At least two active ROIs are required."
        )
        return None, None

    traces_array = np.vstack(traces)  # (n_rois, n_frames)
    dff_zero_mean = zscore(traces_array, axis=1)

    n_rois = len(rois_idxs)
    correlation_matrix_active = np.empty((n_rois, n_rois), dtype=float)

    norms = np.linalg.norm(dff_zero_mean, axis=1)
    norms[norms == 0] = np.finfo(float).eps

    np.fill_diagonal(correlation_matrix_active, 1.0)

    for i in range(n_rois):
        x = dff_zero_mean[i]
        for j in range(i + 1, n_rois):
            y = dff_zero_mean[j]
            corr = correlate(x, y, mode="full", method="fft")
            corr /= norms[i] * norms[j]
            max_corr = float(np.max(corr))
            correlation_matrix_active[i, j] = max_corr
            correlation_matrix_active[j, i] = max_corr

    return correlation_matrix_active, rois_idxs


# -----------------------------------------------------------------------------#
# Plotting with pyqtgraph
# -----------------------------------------------------------------------------#
def _plot_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot the pairwise cross-correlation matrix as a heatmap (pyqtgraph)."""
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()

    # Hide shared legend if present (we don't want it here)
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    correlation_matrix, rois_idxs = _calculate_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        plot.setTitle("Pairwise Cross-Correlation Matrix\n(No data)")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    corr = correlation_matrix
    corr.shape[0]

    # ---------------- IMAGE ITEM (centered, full view) ---------------- #
    img = pg.ImageItem(corr)

    # viridis colormap
    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))  # fixed [0, 1]

    plot.addItem(img)

    # ViewBox & geometry
    vb = plot.getViewBox()

    # Make (0,0) top-left like imshow
    vb.invertY(True)

    # keep it square
    vb.setAspectLocked(True)  # or vb.setAspectLocked(True, ratio=1)

    plot.setTitle("Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events)")
    plot.setLabel("bottom", "ROI index")
    plot.setLabel("left", "ROI index")

    # Hide axis tick labels (like the MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Correlation")

    # ---------------- Hover + Click interaction ---------------- #
    _attach_heatmap_interaction(widget, plot, vb, rois_idxs, corr)


# -----------------------------------------------------------------------------#
# Hover + click helper
# -----------------------------------------------------------------------------#
def _attach_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """
    Attach interaction to the heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with a tuple (roi_i, roi_j)
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # If we reconnect many times, avoid stacking multiple handlers
    old_hover = plot.property("ccorr_hover_handler")
    old_click = plot.property("ccorr_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    base_title = "Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events)"

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
            # Emit tuple; roiSelected is Signal(object), so this is fine
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect on next call
    plot.setProperty("ccorr_hover_handler", _on_mouse_moved)
    plot.setProperty("ccorr_click_handler", _on_mouse_clicked)


def _add_colorbar_to_widget(
    widget: _SingleWellGraphWidget,
    vmin: float,
    vmax: float,
    label: str = "Correlation",
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
