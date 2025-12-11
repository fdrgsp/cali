from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, AnalysisSettings, CaliResult, FOVAnalysis

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Database query for pre-computed spike synchrony matrix
# -----------------------------------------------------------------------------#
def _get_spike_synchrony_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None, float | None, float | None]:
    """Get the pre-computed spike synchrony matrix from database.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    tuple[np.ndarray | None, list[int] | None, float | None, float | None]
        (synchrony_matrix, roi_labels, global_synchrony, jitter_window_ms)
        or (None, None, None, None)
    """
    if run_id is None:
        cali_logger.warning("No run ID specified for spike synchrony plot.")
        return None, None, None, None

    try:
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .where(col(FOV.name) == fov_name)
                .where(col(FOVAnalysis.analysis_result_id) == run_id)
            )

            fov_analysis = session.exec(stmt).first()

            if fov_analysis is None:
                cali_logger.info(
                    f"No FOVAnalysis found for FOV {fov_name} and run {run_id}"
                )
                return None, None, None, None

            if (
                fov_analysis.spike_jitter_synchrony_matrix is None
                or fov_analysis.active_roi_labels is None
            ):
                cali_logger.info(
                    f"FOVAnalysis for {fov_name} has no spike synchrony matrix"
                )
                return None, None, None, None

            # Get jitter window from analysis settings
            cali_result = session.exec(
                select(CaliResult).where(CaliResult.id == run_id)
            ).first()
            jitter_window_ms = None
            if cali_result and cali_result.analysis_settings_id:
                analysis_settings = session.exec(
                    select(AnalysisSettings).where(
                        AnalysisSettings.id == cali_result.analysis_settings_id
                    )
                ).first()
                if analysis_settings:
                    jitter_window_ms = analysis_settings.spikes_sync_cross_corr_lag

            sync_matrix = np.asarray(
                fov_analysis.spike_jitter_synchrony_matrix, dtype=float
            )
            roi_labels = list(fov_analysis.active_roi_labels)
            global_sync = fov_analysis.global_spike_jitter_synchrony

            return sync_matrix, roi_labels, global_sync, jitter_window_ms
    except OperationalError:
        # Table doesn't exist in older databases
        cali_logger.info("FOVAnalysis table not found in database")
        return None, None, None, None


def _filter_matrix_by_rois(
    matrix: np.ndarray,
    roi_labels: list[int],
    selected_rois: list[int] | None,
) -> tuple[np.ndarray, list[int]]:
    """Filter a synchrony matrix to only include selected ROIs."""
    if selected_rois is None:
        return matrix, roi_labels

    indices = []
    filtered_labels = []
    for i, label in enumerate(roi_labels):
        if label in selected_rois:
            indices.append(i)
            filtered_labels.append(label)

    # Return the filtered labels even if there are fewer than 2
    # (the caller should check for insufficient ROIs)
    if len(indices) < 2:
        return matrix[:0, :0], filtered_labels  # Empty matrix with filtered labels

    indices_arr = np.array(indices)
    filtered_matrix = matrix[np.ix_(indices_arr, indices_arr)]

    return filtered_matrix, filtered_labels


# -----------------------------------------------------------------------------#
# Main plotting entry point (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_spike_synchrony_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
) -> None:
    """Plot spike-based synchrony analysis (pyqtgraph heatmap).

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

    # Query pre-computed synchrony matrix from database
    sync_matrix, roi_labels, global_synchrony, jitter_window_ms = (
        _get_spike_synchrony_matrix_from_db(engine, fov_name, run_id)
    )

    if sync_matrix is None or roi_labels is None:
        cali_logger.warning(
            "No spike synchrony data found for this FOV. Ensure analysis has been run."
        )
        plot.setTitle(f"Inferred Spike Synchrony (No data){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter to selected ROIs if specified
    sync, active_roi_ids = _filter_matrix_by_rois(sync_matrix, roi_labels, rois)

    if len(active_roi_ids) < 2:
        cali_logger.warning("Need at least 2 ROIs for synchrony plot.")
        plot.setTitle(f"Inferred Spike Synchrony (Need ≥2 ROIs){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Recalculate global synchrony if ROI subset is selected
    if rois is not None and len(rois) < len(roi_labels):
        n = sync.shape[0]
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            global_synchrony = float(np.median(sync[mask]))
        else:
            global_synchrony = 0.0
    elif global_synchrony is None:
        global_synchrony = 0.0

    # Format jitter window for title
    if jitter_window_ms is not None:
        jitter_str = f"±{jitter_window_ms:.1f}ms"
    else:
        jitter_str = "±window"

    title = (
        "Inferred Spike Synchrony - "
        f"Jitter ({jitter_str}) - "
        f"Global Median: {global_synchrony:.4f}{title_suffix}"
    )

    # ---------------- IMAGE ITEM ---------------- #
    img = pg.ImageItem(sync)

    # viridis colormap
    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))  # fixed [0, 1]

    plot.addItem(img)

    # ViewBox & geometry
    vb = plot.getViewBox()
    vb.invertY(True)  # make (0,0) top-left like imshow
    vb.setAspectLocked(True)  # keep matrix square
    vb.enableAutoRange(x=True, y=True)

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    # Hide axis tick labels (to match MPL style)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Synchrony")

    # ---------------- Hover + Click interaction ---------------- #
    _attach_spike_sync_interaction(widget, plot, vb, active_roi_ids, sync)


# -----------------------------------------------------------------------------#
# Interaction (hover + click)
# -----------------------------------------------------------------------------#
def _attach_spike_sync_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """
    Attach interaction to the spike synchrony heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with [roi_i, roi_j] (as strings)
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # Avoid stacking multiple handlers on repeated calls
    old_hover = plot.property("spike_sync_hover_handler")
    old_click = plot.property("spike_sync_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    base_title = plot.titleLabel.text if plot.titleLabel is not None else ""

    def _on_mouse_moved(pos: pg.QtCore.QPointF) -> None:
        if not plot.sceneBoundingRect().contains(pos):
            plot.setTitle(base_title)
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            val = float(values[row, col])
            plot.setTitle(f"{base_title} | ROI {roi_i} vs ROI {roi_j}: {val:.3f}")
        else:
            plot.setTitle(base_title)

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            # Emit list[str] to match your existing roiSelected semantics
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect next time
    plot.setProperty("spike_sync_hover_handler", _on_mouse_moved)
    plot.setProperty("spike_sync_click_handler", _on_mouse_clicked)


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
