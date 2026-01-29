"""Plot zero-lag Pearson correlation matrices for DF/F and deconvolved DF/F traces."""

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


def _get_dff_correlation_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Get the pre-computed DF/F correlation matrix from database.

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
    tuple[np.ndarray | None, list[int] | None]
        (correlation_matrix, roi_labels) or (None, None) if not found
    """
    if run_id is None:
        cali_logger.warning("No run ID specified for DF/F correlation plot.")
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

            if (
                fov_analysis.calcium_dff_correlation_matrix is None
                or fov_analysis.active_roi_labels is None
            ):
                cali_logger.info(
                    f"FOVAnalysis for {fov_name} has no DF/F correlation matrix"
                )
                return None, None

            corr_matrix = np.asarray(
                fov_analysis.calcium_dff_correlation_matrix, dtype=float
            )
            roi_labels = list(fov_analysis.active_roi_labels)

            return corr_matrix, roi_labels
    except OperationalError:
        cali_logger.info("FOVAnalysis table not found in database")
        return None, None


def _get_dec_dff_correlation_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Get the pre-computed deconvolved DF/F correlation matrix from database.

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
    tuple[np.ndarray | None, list[int] | None]
        (correlation_matrix, roi_labels) or (None, None) if not found
    """
    if run_id is None:
        cali_logger.warning(
            "No run ID specified for deconvolved DF/F correlation plot."
        )
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

            if (
                fov_analysis.calcium_dec_dff_corr_matrix is None
                or fov_analysis.active_roi_labels is None
            ):
                cali_logger.info(
                    f"FOVAnalysis for {fov_name} "
                    f"has no deconvolved DF/F correlation matrix"
                )
                return None, None

            corr_matrix = np.asarray(
                fov_analysis.calcium_dec_dff_corr_matrix, dtype=float
            )
            roi_labels = list(fov_analysis.active_roi_labels)

            return corr_matrix, roi_labels
    except OperationalError:
        cali_logger.info("FOVAnalysis table not found in database")
        return None, None


def _filter_matrix_by_rois(
    matrix: np.ndarray,
    roi_labels: list[int],
    selected_rois: list[int] | None,
) -> tuple[np.ndarray, list[int]]:
    """Filter a correlation matrix to only include selected ROIs.

    Parameters
    ----------
    matrix : np.ndarray
        Full NxN matrix
    roi_labels : list[int]
        ROI labels corresponding to matrix indices
    selected_rois : list[int] | None
        ROI labels to include, or None for all

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (filtered_matrix, filtered_roi_labels)
    """
    if selected_rois is None:
        return matrix, roi_labels

    # Find indices of selected ROIs
    indices = [i for i, label in enumerate(roi_labels) if label in selected_rois]

    if not indices:
        return np.array([]), []

    # Filter matrix
    filtered_matrix = matrix[np.ix_(indices, indices)]
    filtered_labels = [roi_labels[i] for i in indices]

    return filtered_matrix, filtered_labels


def _plot_dff_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
) -> None:
    """Plot zero-lag Pearson correlation on DF/F traces.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Plot widget
    engine : Engine
        Database engine
    fov_name : str
        FOV name
    rois : list[int] | None
        Selected ROI labels, or None for all
    run_id : int | None
        Analysis run ID
    title_suffix : str
        Optional suffix to add to plot titles (e.g., " (Stimulated)")
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    correlation_matrix, roi_labels = _get_dff_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if correlation_matrix is None or roi_labels is None:
        plot.setTitle(f"Pairwise Pearson Correlation (No data){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    corr, rois_idxs = _filter_matrix_by_rois(correlation_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        plot.setTitle(f"Pairwise Pearson Correlation (Need ≥2 ROIs){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    img = pg.ImageItem(corr)
    img.setLookupTable(CMAP.getLookupTable(0, 1, 256))
    img.setLevels((-1.0, 1.0))

    plot.addItem(img)

    vb = plot.getViewBox()
    vb.invertY(True)
    vb.setAspectLocked(True)

    # Calculate median of off-diagonal elements
    mask = ~np.eye(corr.shape[0], dtype=bool)
    median_corr = np.median(corr[mask])

    title = (
        f"Pairwise Pearson Correlation (Zero-lag - ΔF/F Traces) "
        f"(median: {median_corr:.3f}){title_suffix}"
    )
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    _attach_heatmap_interaction(widget, plot, vb, rois_idxs, corr, title)


def _plot_dec_dff_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
) -> None:
    """Plot zero-lag Pearson correlation on deconvolved DF/F traces.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Plot widget
    engine : Engine
        Database engine
    fov_name : str
        FOV name
    rois : list[int] | None
        Selected ROI labels, or None for all
    run_id : int | None
        Analysis run ID
    title_suffix : str
        Optional suffix to add to plot titles (e.g., " (Stimulated)")
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    correlation_matrix, roi_labels = _get_dec_dff_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if correlation_matrix is None or roi_labels is None:
        plot.setTitle(f"Pairwise Pearson Correlation (No data){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    corr, rois_idxs = _filter_matrix_by_rois(correlation_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        plot.setTitle(f"Pairwise Pearson Correlation (Need ≥2 ROIs){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    img = pg.ImageItem(corr)
    img.setLookupTable(CMAP.getLookupTable(0, 1, 256))
    img.setLevels((-1.0, 1.0))

    plot.addItem(img)

    vb = plot.getViewBox()
    vb.invertY(True)
    vb.setAspectLocked(True)

    # Calculate median of off-diagonal elements
    mask = ~np.eye(corr.shape[0], dtype=bool)
    median_corr = np.median(corr[mask])

    title = (
        "Pairwise Pearson Correlation (Zero-Lag - "
        f"Deconvolved ΔF/F Traces) (median: {median_corr:.3f}){title_suffix}"
    )
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    _attach_heatmap_interaction(widget, plot, vb, rois_idxs, corr, title)


def _attach_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    vb: pg.ViewBox,
    roi_labels: list[int],
    matrix: np.ndarray,
    base_title: str,
) -> None:
    """Attach hover and click interaction to correlation heatmap.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Plot widget
    plot : pg.PlotItem
        Plot item
    vb : pg.ViewBox
        ViewBox
    roi_labels : list[int]
        ROI labels
    matrix : np.ndarray
        Correlation matrix
    base_title : str
        Base title for the plot
    """
    n_rows, n_cols = matrix.shape
    scene = plot.scene()

    # Disconnect old handlers if they exist
    old_hover = plot.property("dff_corr_hover_handler")
    old_click = plot.property("dff_corr_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    def _on_mouse_moved(pos: pg.Point) -> None:
        """Show correlation value on hover."""
        if not plot.sceneBoundingRect().contains(pos):
            plot.setTitle(base_title)
            return

        mouse_point = vb.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())

        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = roi_labels[row]
            roi_j = roi_labels[col]
            val = float(matrix[row, col])
            plot.setTitle(f"{base_title} | ROI {roi_i} vs ROI {roi_j}: r = {val:.3f}")
        else:
            plot.setTitle(base_title)

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        """Handle click events and emit roiSelected signal."""
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = roi_labels[row]
            roi_j = roi_labels[col]
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    # Connect handlers
    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect on next call
    plot.setProperty("dff_corr_hover_handler", _on_mouse_moved)
    plot.setProperty("dff_corr_click_handler", _on_mouse_clicked)
