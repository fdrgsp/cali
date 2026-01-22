from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import add_colorbar_to_widget, disconnect_hover_handlers
from cali.sqlmodel._model import FOV, AnalysisSettings, CaliResult, FOVAnalysis

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# PLOT STYLE CONSTANTS
CMAP_NAME = "CET-D1A"
CMAP = pg.colormap.get(CMAP_NAME)


# -----------------------------------------------------------------------------#
# Database query for pre-computed spike max-lag values matrix
# -----------------------------------------------------------------------------#
def _get_spike_max_lag_values_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
    rising_edges: bool = False,
) -> tuple[np.ndarray | None, list[int] | None, int | None]:
    """Get the pre-computed spike max-lag values matrix from database.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    run_id : int | None
        Filter by specific analysis run
    rising_edges : bool
        If True, use rising_edges matrix; otherwise use thresholded binary matrix

    Returns
    -------
    tuple[np.ndarray | None, list[int] | None, int | None]
        (lag_matrix, roi_labels, max_lag_frames) or (None, None, None) if not found
    """
    if run_id is None:
        cali_logger.warning("No run ID specified for spike max-lag values plot.")
        return None, None, None

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
                return None, None, None

            # Get the appropriate matrix based on rising_edges parameter
            if rising_edges:
                lag_matrix_data = fov_analysis.spike_max_lag_values_matrix_rising_edges
                matrix_type = "rising edges"
            else:
                lag_matrix_data = fov_analysis.spike_max_lag_values_matrix
                matrix_type = "thresholded binary"

            if lag_matrix_data is None or fov_analysis.active_roi_labels is None:
                cali_logger.info(
                    f"FOVAnalysis for {fov_name} has no spike max-lag values "
                    f"matrix ({matrix_type})"
                )
                return None, None, None

            lag_matrix = np.asarray(lag_matrix_data, dtype=int)
            roi_labels = list(fov_analysis.active_roi_labels)

            # Get max_lag from analysis settings
            max_lag_frames = None
            cali_result = session.exec(
                select(CaliResult).where(CaliResult.id == run_id)
            ).first()
            if cali_result and cali_result.analysis_settings_id:
                analysis_settings = session.exec(
                    select(AnalysisSettings).where(
                        AnalysisSettings.id == cali_result.analysis_settings_id
                    )
                ).first()
                if analysis_settings:
                    # Convert from ms to frames
                    max_lag_ms = analysis_settings.spikes_sync_cross_corr_lag
                    frame_rate = analysis_settings.frame_rate
                    max_lag_frames = int(max_lag_ms * frame_rate / 1000.0)

            return lag_matrix, roi_labels, max_lag_frames
    except OperationalError:
        # Table doesn't exist in older databases
        cali_logger.info("FOVAnalysis table not found in database")
        return None, None, None


def _filter_matrix_by_rois(
    matrix: np.ndarray,
    roi_labels: list[int],
    selected_rois: list[int] | None,
) -> tuple[np.ndarray, list[int]]:
    """Filter a lag matrix to only include selected ROIs.

    Parameters
    ----------
    matrix : np.ndarray
        Full NxN matrix
    roi_labels : list[int]
        ROI labels corresponding to matrix indices
    selected_rois : list[int] | None
        ROIs to filter to, or None to keep all

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (filtered_matrix, filtered_roi_labels)
    """
    if selected_rois is None:
        return matrix, roi_labels

    # Find indices of selected ROIs in the full matrix
    indices = []
    filtered_labels = []
    for i, label in enumerate(roi_labels):
        if label in selected_rois:
            indices.append(i)
            filtered_labels.append(label)

    if len(indices) < 2:
        return matrix, roi_labels  # Return full matrix if too few ROIs selected

    # Extract submatrix
    indices_arr = np.array(indices)
    filtered_matrix = matrix[np.ix_(indices_arr, indices_arr)]

    return filtered_matrix, filtered_labels


# -----------------------------------------------------------------------------#
# Plotting with pyqtgraph
# -----------------------------------------------------------------------------#
def _plot_spike_max_lag_values_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
    rising_edges: bool = False,
) -> None:
    """Plot the spike max-lag values as a heatmap (pyqtgraph).

    Shows the lag (in frames) at which maximum cross-correlation occurs for each
    pair of neurons. Positive lag means the second neuron lags behind the first
    (i.e., the first neuron leads). Negative lag means the first neuron lags behind
    the second (i.e., the second neuron leads).

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        ROI labels to include, or None for all
    run_id : int | None
        Analysis run ID
    title_suffix : str
        Optional suffix to add to plot titles (e.g., " - Stimulated")
    rising_edges : bool
        If True, use rising edge spike data; otherwise use thresholded binary
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    # Hide shared legend if present (we don't want it here)
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Query pre-computed lag matrix from database
    (
        lag_matrix,
        roi_labels,
        max_lag_frames,
    ) = _get_spike_max_lag_values_matrix_from_db(
        engine, fov_name, run_id, rising_edges=rising_edges
    )

    if lag_matrix is None or roi_labels is None:
        plot.setTitle(f"Inferred Spikes Max-Lag Values (No data){title_suffix}")
        plot.setLabel("bottom", "ROI (j)")
        plot.setLabel("left", "ROI (i)")
        return

    # Filter to selected ROIs if specified
    lags, rois_idxs = _filter_matrix_by_rois(lag_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        plot.setTitle(f"Inferred Spikes Max-Lag Values (Need ≥2 ROIs){title_suffix}")
        plot.setLabel("bottom", "ROI (j)")
        plot.setLabel("left", "ROI (i)")
        return

    # ---------------- IMAGE ITEM (centered, full view) ---------------- #
    img = pg.ImageItem(lags.astype(float))

    # Determine color scale limits
    if max_lag_frames is not None:
        vmin, vmax = -max_lag_frames, max_lag_frames
    else:
        abs_max = max(abs(lags.min()), abs(lags.max()))
        vmin, vmax = -abs_max, abs_max

    # getLookupTable takes colormap positions (0-1), not data values
    # setLevels maps data values to the colormap
    img.setLookupTable(CMAP.getLookupTable(0, 1, 256))
    img.setLevels((vmin, vmax))

    plot.addItem(img)

    # ViewBox & geometry
    vb = plot.getViewBox()

    # Make (0,0) top-left like imshow
    vb.invertY(True)

    # keep it square
    vb.setAspectLocked(True)

    spike_type = "Rising Edges" if rising_edges else "Thresholded"
    title = f"Inferred Spikes Max-Lag Values ({spike_type}) (frames){title_suffix}"
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI (j)")
    plot.setLabel("left", "ROI (i)")

    # Hide axis tick labels (like the MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    add_colorbar_to_widget(
        widget,
        vmin=vmin,
        vmax=vmax,
        label="Lag (frames)\n←j leads | i leads→",
        colormap=CMAP_NAME,
    )

    # ---------------- Hover + Click interaction ---------------- #
    _attach_heatmap_interaction(widget, plot, vb, rois_idxs, lags, title)


# -----------------------------------------------------------------------------#
# Hover + click helper
# -----------------------------------------------------------------------------#
def _attach_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
    base_title: str = "",
) -> None:
    """
    Attach interaction to the heatmap.

    - Hover: show ROI_i, ROI_j, lag value in the title
    - Click: emit widget.roiSelected with a tuple (roi_i, roi_j)
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # If we reconnect many times, avoid stacking multiple handlers
    old_hover = plot.property("spike_maxlag_values_hover_handler")
    old_click = plot.property("spike_maxlag_values_click_handler")
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
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            lag = int(values[row, col])
            if lag > 0:
                msg = (
                    f"{base_title} | ROI {roi_i} → ROI {roi_j}: +{lag} frames (j lags)"
                )
            elif lag < 0:
                msg = (
                    f"{base_title} | ROI {roi_i} ← ROI {roi_j}: {lag} frames (j leads)"
                )
            else:
                msg = f"{base_title} | ROI {roi_i} ↔ ROI {roi_j}: {lag} frames (sync)"
            plot.setTitle(msg)
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
            # Emit tuple; roiSelected is Signal(object), so this is fine
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect on next call
    plot.setProperty("spike_maxlag_values_hover_handler", _on_mouse_moved)
    plot.setProperty("spike_maxlag_values_click_handler", _on_mouse_clicked)
