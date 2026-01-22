from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import add_colorbar_to_widget, disconnect_hover_handlers
from cali.sqlmodel._model import FOV, FOVAnalysis

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# PLOT STYLE CONSTANTS
CMAP_NAME = "viridis"
CMAP = pg.colormap.get(CMAP_NAME)


# -----------------------------------------------------------------------------#
# Database query for pre-computed spike max-lag correlation matrix
# -----------------------------------------------------------------------------#
def _get_spike_max_lag_correlation_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
    rising_edges: bool = False,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Get the pre-computed spike max-lag correlation matrix from database.

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
    tuple[np.ndarray | None, list[int] | None]
        (correlation_matrix, roi_labels) or (None, None) if not found
    """
    if run_id is None:
        cali_logger.warning("No run ID specified for spike max-lag correlation plot.")
        return None, None

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
                return None, None

            # Get the appropriate matrix based on rising_edges parameter
            if rising_edges:
                corr_matrix_data = (
                    fov_analysis.spike_max_lag_correlation_matrix_rising_edges
                )
                matrix_type = "rising edges"
            else:
                corr_matrix_data = fov_analysis.spike_max_lag_correlation_matrix
                matrix_type = "thresholded binary"

            if corr_matrix_data is None or fov_analysis.active_roi_labels is None:
                cali_logger.info(
                    f"FOVAnalysis for {fov_name} has no spike max-lag "
                    f"correlation matrix ({matrix_type})"
                )
                return None, None

            corr_matrix = np.asarray(corr_matrix_data, dtype=float)
            roi_labels = list(fov_analysis.active_roi_labels)

            return corr_matrix, roi_labels
    except OperationalError:
        # Table doesn't exist in older databases
        cali_logger.info("FOVAnalysis table not found in database")
        return None, None


def _filter_matrix_by_rois(
    matrix: np.ndarray,
    roi_labels: list[int],
    selected_rois: list[int] | None,
) -> tuple[np.ndarray, list[int]]:
    """Filter a correlation/synchrony matrix to only include selected ROIs.

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
def _plot_spike_max_lag_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
    rising_edges: bool = False,
) -> None:
    """Plot the spike max-lag cross-correlation matrix as a heatmap (pyqtgraph).

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

    # Query pre-computed correlation matrix from database
    correlation_matrix, roi_labels = _get_spike_max_lag_correlation_matrix_from_db(
        engine, fov_name, run_id, rising_edges=rising_edges
    )

    if correlation_matrix is None or roi_labels is None:
        plot.setTitle(
            f"Inferred Spikes Peak CCG at Optimal Lag (No data){title_suffix}"
        )
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter to selected ROIs if specified
    corr, rois_idxs = _filter_matrix_by_rois(correlation_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        plot.setTitle(
            f"Inferred Spikes Peak CCG at Optimal Lag (Need ≥2 ROIs){title_suffix}"
        )
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # ---------------- IMAGE ITEM (centered, full view) ---------------- #
    img = pg.ImageItem(corr)

    # Per-trigger probability normalization: values are typically in [0, ~0.5]
    # but can exceed 1.0 in edge cases, so we use data-driven limits
    vmin = 0.0
    vmax = max(1.0, float(np.nanmax(corr)))  # At least 0-1, or higher if needed

    img.setLookupTable(CMAP.getLookupTable(0, 1, 256))
    img.setLevels((vmin, vmax))

    plot.addItem(img)

    # ViewBox & geometry
    vb = plot.getViewBox()

    # Make (0,0) top-left like imshow
    vb.invertY(True)

    # keep it square
    vb.setAspectLocked(True)

    # Calculate median of off-diagonal elements
    mask = ~np.eye(corr.shape[0], dtype=bool)
    median_corr = np.median(corr[mask])

    spike_type = "Rising Edges" if rising_edges else "Thresholded"
    title = (
        f"Inferred Spikes Peak CCG at Optimal Lag ({spike_type}) "
        f"(median: {median_corr:.3f}){title_suffix}"
    )
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    # Hide axis tick labels (like the MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar - label reflects per-trigger probability normalization
    add_colorbar_to_widget(
        widget,
        vmin=vmin,
        vmax=vmax,
        label="P(spike|ref)",  # Per-trigger probability
        colormap=CMAP_NAME,
    )

    # ---------------- Hover + Click interaction ---------------- #
    _attach_heatmap_interaction(widget, plot, vb, rois_idxs, corr, title)


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

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with a tuple (roi_i, roi_j)
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # If we reconnect many times, avoid stacking multiple handlers
    old_hover = plot.property("spike_maxlag_hover_handler")
    old_click = plot.property("spike_maxlag_click_handler")
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
            # Emit tuple; roiSelected is Signal(object), so this is fine
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect on next call
    plot.setProperty("spike_maxlag_hover_handler", _on_mouse_moved)
    plot.setProperty("spike_maxlag_click_handler", _on_mouse_clicked)


# -----------------------------------------------------------------------------#
# Database query for pre-computed CCG z-score matrix
# -----------------------------------------------------------------------------#
def _get_ccg_zscore_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
    rising_edges: bool = False,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Get the pre-computed CCG z-score matrix from database.

    The z-score matrix contains significance values for the CCG at each
    ROI pair's optimal lag. Z-scores are computed using shift predictor
    baseline correction: z = (CCG_raw - baseline_mean) / baseline_std

    |z| > 2 suggests significant functional connectivity.

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
    tuple[np.ndarray | None, list[int] | None]
        (zscore_matrix, roi_labels) or (None, None) if not found
    """
    if run_id is None:
        cali_logger.warning("No run ID specified for CCG z-score plot.")
        return None, None

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
                return None, None

            # Get the appropriate matrix based on rising_edges parameter
            if rising_edges:
                zscore_matrix_data = fov_analysis.spike_ccg_zscore_matrix_rising_edges
                matrix_type = "rising edges"
            else:
                zscore_matrix_data = fov_analysis.spike_ccg_zscore_matrix
                matrix_type = "thresholded binary"

            if zscore_matrix_data is None or fov_analysis.active_roi_labels is None:
                cali_logger.info(
                    f"FOVAnalysis for {fov_name} has no CCG z-score "
                    f"matrix ({matrix_type})"
                )
                return None, None

            zscore_matrix = np.asarray(zscore_matrix_data, dtype=float)
            roi_labels = list(fov_analysis.active_roi_labels)

            return zscore_matrix, roi_labels
    except OperationalError:
        # Table doesn't exist in older databases
        cali_logger.info("FOVAnalysis table not found in database")
        return None, None


# -----------------------------------------------------------------------------#
# Plotting CCG Z-Score Matrix
# -----------------------------------------------------------------------------#
ZSCORE_CMAP_NAME = "CET-D1A"  # Diverging colormap for z-scores
ZSCORE_CMAP = pg.colormap.get(ZSCORE_CMAP_NAME)


def _plot_ccg_zscore_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
    rising_edges: bool = False,
) -> None:
    """Plot the CCG z-score matrix as a heatmap (pyqtgraph).

    Z-scores indicate statistical significance of the CCG values relative
    to a shuffled baseline. |z| > 2 suggests significant functional
    connectivity between the ROI pair.

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
        Optional suffix to add to plot titles
    rising_edges : bool
        If True, use rising edge spike data; otherwise use thresholded binary
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Query pre-computed z-score matrix from database
    zscore_matrix, roi_labels = _get_ccg_zscore_matrix_from_db(
        engine, fov_name, run_id, rising_edges=rising_edges
    )

    if zscore_matrix is None or roi_labels is None:
        plot.setTitle(f"Inferred Spikes CCG Z-Score (No data){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter to selected ROIs if specified
    zscores, rois_idxs = _filter_matrix_by_rois(zscore_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        plot.setTitle(f"Inferred Spikes CCG Z-Score (Need ≥2 ROIs){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Handle inf values on diagonal (self-correlation)
    # Replace inf with NaN for display purposes
    zscores_display = zscores.copy()
    zscores_display[np.isinf(zscores_display)] = np.nan

    # ---------------- IMAGE ITEM ---------------- #
    img = pg.ImageItem(zscores_display)

    # Symmetric color scale centered at 0
    # Typical z-score range: [-4, 4] but use data-driven limits
    finite_vals = zscores_display[np.isfinite(zscores_display)]
    if len(finite_vals) > 0:
        abs_max = max(4.0, np.abs(finite_vals).max())  # At least ±4
    else:
        abs_max = 4.0
    vmin, vmax = -abs_max, abs_max

    img.setLookupTable(ZSCORE_CMAP.getLookupTable(0, 1, 256))
    img.setLevels((vmin, vmax))

    plot.addItem(img)

    # ViewBox & geometry
    vb = plot.getViewBox()
    vb.invertY(True)
    vb.setAspectLocked(True)

    # Calculate statistics (excluding diagonal/inf values)
    mask = ~np.eye(zscores.shape[0], dtype=bool) & np.isfinite(zscores)
    if np.any(mask):
        median_z = np.median(zscores[mask])
        # Count significant pairs (|z| > 2)
        n_significant = np.sum(np.abs(zscores[mask]) > 2)
        n_pairs = np.sum(mask)
        pct_significant = 100.0 * n_significant / n_pairs if n_pairs > 0 else 0.0
    else:
        median_z = 0.0
        pct_significant = 0.0

    spike_type = "Rising Edges" if rising_edges else "Thresholded"
    title = (
        f"Inferred Spikes CCG Z-Score ({spike_type}) "
        f"(median: {median_z:.2f}, {pct_significant:.1f}% significant){title_suffix}"
    )
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    # Hide axis tick labels
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar with significance reference
    add_colorbar_to_widget(
        widget,
        vmin=vmin,
        vmax=vmax,
        label="Z-score\n(|z|>2: significant)",
        colormap=ZSCORE_CMAP_NAME,
    )

    # ---------------- Hover + Click interaction ---------------- #
    _attach_zscore_heatmap_interaction(widget, plot, vb, rois_idxs, zscores, title)


def _attach_zscore_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
    base_title: str = "",
) -> None:
    """Attach interaction to the z-score heatmap.

    - Hover: show ROI_i, ROI_j, z-score and significance in the title
    - Click: emit widget.roiSelected with a tuple (roi_i, roi_j)
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # Cleanup existing handlers
    old_hover = plot.property("ccg_zscore_hover_handler")
    old_click = plot.property("ccg_zscore_click_handler")
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
            z = float(values[row, col])
            if np.isinf(z):
                plot.setTitle(f"{base_title} | ROI {roi_i} vs {roi_j}: self")
            elif abs(z) > 2:
                plot.setTitle(
                    f"{base_title} | ROI {roi_i} vs {roi_j}: z={z:.2f} (significant)"
                )
            else:
                plot.setTitle(f"{base_title} | ROI {roi_i} vs {roi_j}: z={z:.2f}")
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
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers
    plot.setProperty("ccg_zscore_hover_handler", _on_mouse_moved)
    plot.setProperty("ccg_zscore_click_handler", _on_mouse_clicked)
