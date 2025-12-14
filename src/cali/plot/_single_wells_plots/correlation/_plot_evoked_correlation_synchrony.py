"""Evoked experiment correlation and synchrony plots.

Plots for stimulated vs non-stimulated ROIs. These wrappers filter ROIs
by stimulation status before calling the standard correlation and synchrony
plotting functions.

Also includes sorted combined plots where stimulated ROIs are shown first,
followed by non-stimulated ROIs, to visualize network clustering.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.plot._util import add_colorbar_to_widget, disconnect_hover_handlers
from cali.sqlmodel._model import FOV, ROI

from ._plot_calcium_traces_correlation import (
    _get_dec_dff_correlation_matrix_from_db,
)
from ._plot_inferred_spike_correlation import (
    _get_spike_correlation_matrix_from_db,
    _plot_spike_correlation_data,
)
from ._plot_inferred_spike_synchrony import (
    _get_spike_synchrony_matrix_from_db,
    _plot_spike_synchrony_data,
)
from ._plot_spike_max_lag_correlation import (
    _get_spike_max_lag_correlation_matrix_from_db,
    _plot_spike_max_lag_correlation_data,
)

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# PLOT STYLE CONSTANTS
STIM_RECTANGLE_COLOR = "orange"
STIM_RECTANGLE_WIDTH = 8
CMAP_NAME = "viridis"
CMAP = pg.colormap.get(CMAP_NAME)


def _filter_rois_by_stimulation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None,
    stimulated: bool,
) -> list[int] | None:
    """Filter ROIs by stimulation status.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        FOV name to query
    rois : list[int] | None
        Initial ROI filter (None for all ROIs in FOV)
    stimulated : bool
        If True, return only stimulated ROIs. If False, return only non-stimulated.

    Returns
    -------
    list[int] | None
        Filtered list of ROI label_values, or None if no ROIs match
    """
    with Session(engine) as session:
        stmt = (
            select(ROI.label_value)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.stimulated) == stimulated)  # Filter by stimulation status
            .where(col(ROI.active) == True)  # noqa: E712
        )

        # Apply user ROI filter if provided
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        filtered_rois = list(session.exec(stmt).all())

    return filtered_rois if filtered_rois else None


# =============================================================================
# Inferred Spikes - Stimulated ROIs
# =============================================================================


def _plot_stimulated_spike_synchrony(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes synchrony for stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(engine, fov_name, rois, stimulated=True)
    _plot_spike_synchrony_data(
        widget, engine, fov_name, filtered_rois, run_id, title_suffix=" (Stimulated)"
    )


def _plot_stimulated_spike_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes cross-correlation for stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(engine, fov_name, rois, stimulated=True)
    _plot_spike_correlation_data(
        widget, engine, fov_name, filtered_rois, run_id, title_suffix=" (Stimulated)"
    )


def _plot_stimulated_spike_max_lag_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes max-lag cross-correlation for stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(engine, fov_name, rois, stimulated=True)
    _plot_spike_max_lag_correlation_data(
        widget, engine, fov_name, filtered_rois, run_id, title_suffix=" (Stimulated)"
    )


# =============================================================================
# Inferred Spikes - Non-Stimulated ROIs
# =============================================================================


def _plot_non_stimulated_spike_synchrony(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes synchrony for non-stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(
        engine, fov_name, rois, stimulated=False
    )
    _plot_spike_synchrony_data(
        widget,
        engine,
        fov_name,
        filtered_rois,
        run_id,
        title_suffix=" (Non-Stimulated)",
    )


def _plot_non_stimulated_spike_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes cross-correlation for non-stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(
        engine, fov_name, rois, stimulated=False
    )
    _plot_spike_correlation_data(
        widget,
        engine,
        fov_name,
        filtered_rois,
        run_id,
        title_suffix=" (Non-Stimulated)",
    )


def _plot_non_stimulated_spike_max_lag_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes max-lag cross-correlation for non-stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(
        engine, fov_name, rois, stimulated=False
    )
    _plot_spike_max_lag_correlation_data(
        widget,
        engine,
        fov_name,
        filtered_rois,
        run_id,
        title_suffix=" (Non-Stimulated)",
    )


# =============================================================================
# Helper functions for sorted combined plots
# =============================================================================


def _detach_heatmap_interaction(plot: pg.PlotItem) -> None:
    """Detach any existing hover and click handlers from a heatmap plot.

    Parameters
    ----------
    plot : pg.PlotItem
        Plot item to clean up
    """
    old_hover = plot.property("evoked_hover_handler")
    old_click = plot.property("evoked_click_handler")

    # Disconnect from scene signals if scene exists
    scene = plot.scene()
    if scene is not None:
        if old_hover is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                scene.sigMouseMoved.disconnect(old_hover)

        if old_click is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                scene.sigMouseClicked.disconnect(old_click)

    # Always clear the property references, even if no scene
    if old_hover is not None:
        plot.setProperty("evoked_hover_handler", None)
    if old_click is not None:
        plot.setProperty("evoked_click_handler", None)


def _attach_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    base_title: str,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """Attach hover and click interaction to a heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with a tuple (roi_i, roi_j)
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # Clean up any existing handlers first
    _detach_heatmap_interaction(plot)

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
    plot.setProperty("evoked_hover_handler", _on_mouse_moved)
    plot.setProperty("evoked_click_handler", _on_mouse_clicked)


def _get_sorted_rois_by_stimulation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None,
) -> tuple[list[int], list[int], list[int]]:
    """Get ROIs sorted by stimulation status (stimulated first, then non-stimulated).

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        FOV name to query
    rois : list[int] | None
        Initial ROI filter (None for all ROIs in FOV)

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        (all_sorted_rois, stimulated_rois, non_stimulated_rois)
        all_sorted_rois is the concatenation of stimulated + non-stimulated
    """
    with Session(engine) as session:
        # Get stimulated ROIs
        stmt_stim = (
            select(ROI.label_value)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.stimulated) == True)  # noqa: E712
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt_stim = stmt_stim.where(col(ROI.label_value).in_(rois))

        stimulated_rois = sorted(session.exec(stmt_stim).all())

        # Get non-stimulated ROIs
        stmt_non_stim = (
            select(ROI.label_value)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.stimulated) == False)  # noqa: E712
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt_non_stim = stmt_non_stim.where(col(ROI.label_value).in_(rois))

        non_stimulated_rois = sorted(session.exec(stmt_non_stim).all())

    # Concatenate: stimulated first, then non-stimulated
    all_sorted_rois = stimulated_rois + non_stimulated_rois

    return all_sorted_rois, stimulated_rois, non_stimulated_rois


def _reorder_matrix_by_roi_list(
    matrix: np.ndarray,
    original_roi_labels: list[int],
    desired_roi_order: list[int],
) -> tuple[np.ndarray | None, list[int]]:
    """Reorder a correlation/synchrony matrix according to a desired ROI ordering.

    Parameters
    ----------
    matrix : np.ndarray
        Original matrix (n_rois x n_rois)
    original_roi_labels : list[int]
        Original ROI labels corresponding to matrix rows/cols
    desired_roi_order : list[int]
        Desired ROI ordering (stimulated first, then non-stimulated)

    Returns
    -------
    tuple[np.ndarray | None, list[int]]
        (reordered_matrix, filtered_roi_labels) or (None, []) if incompatible
    """
    # Find which desired ROIs are actually in the matrix
    available_rois = [roi for roi in desired_roi_order if roi in original_roi_labels]

    if len(available_rois) < 2:
        return None, []

    # Get indices in original matrix for the available ROIs
    indices = [original_roi_labels.index(roi) for roi in available_rois]

    # Reorder both rows and columns
    reordered = matrix[np.ix_(indices, indices)]

    return reordered, available_rois


# =============================================================================
# Inferred Spikes - Sorted Combined Plots (Stimulated → Non-Stimulated)
# =============================================================================


def _plot_sorted_spike_synchrony(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot spike synchrony with ROIs sorted by stimulation status.

    ROIs are ordered: stimulated neurons first, then non-stimulated neurons.
    This reveals network clustering based on stimulation.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    _detach_heatmap_interaction(plot)
    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle("Spike Synchrony (Sorted - Need ≥2 ROIs)")
        return

    # Get synchrony matrix from database
    (
        sync_matrix,
        roi_labels,
        _,  # global_sync not used - we calculate from filtered matrix
        jitter_ms,
    ) = _get_spike_synchrony_matrix_from_db(engine, fov_name, run_id)

    if sync_matrix is None or roi_labels is None:
        plot.setTitle("Spike Synchrony (Sorted - No data)")
        return

    # Reorder matrix according to sorted ROIs
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        sync_matrix, roi_labels, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Spike Synchrony (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = f"Spike Synchrony (Sorted: {n_stim} Stim, {n_non_stim} Non-Stim)"
    if jitter_ms is not None:
        title += f" | Jitter: {jitter_ms}ms"

    # Add medians (always use filtered matrix values, not database global_sync)
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=0.0, vmax=1.0, label="Synchrony", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block (rectangle)
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)

        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)

    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)


def _plot_sorted_spike_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot spike correlation with ROIs sorted by stimulation status.

    ROIs are ordered: stimulated neurons first, then non-stimulated neurons.
    This reveals network clustering based on stimulation.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    _detach_heatmap_interaction(plot)
    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle("Spike Correlation (Sorted - Need ≥2 ROIs)")
        return

    # Get correlation matrix from database
    corr_matrix, roi_labels = _get_spike_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if corr_matrix is None or roi_labels is None:
        plot.setTitle("Spike Correlation (Sorted - No data)")
        return

    # Reorder matrix according to sorted ROIs
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        corr_matrix, roi_labels, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Spike Correlation (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(-1.0, 1.0, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = f"Spike Correlation (Sorted: {n_stim} Stim, {n_non_stim} Non-Stim)"

    # Add medians
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block (green rectangle)
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)
        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)
    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)


def _plot_sorted_spike_max_lag_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot spike max-lag correlation with ROIs sorted by stimulation status.

    ROIs are ordered: stimulated neurons first, then non-stimulated neurons.
    This reveals network clustering based on stimulation.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    _detach_heatmap_interaction(plot)
    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle("Spike Max-Lag Correlation (Sorted - Need ≥2 ROIs)")
        return

    # Get correlation matrix from database
    corr_matrix, roi_labels = _get_spike_max_lag_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if corr_matrix is None or roi_labels is None:
        plot.setTitle("Spike Max-Lag Correlation (Sorted - No data)")
        return

    # Reorder matrix according to sorted ROIs
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        corr_matrix, roi_labels, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Spike Max-Lag Correlation (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = f"Spike Max-Lag Correlation (Sorted: {n_stim} Stim, {n_non_stim} Non-Stim)"

    # Add medians
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=0.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block (green rectangle)
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)

        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)

    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)


def _plot_sorted_dec_dff_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot zero-lag dec DF/F correlation with ROIs sorted by stimulation status.

    ROIs are ordered: stimulated neurons first, then non-stimulated neurons.
    This reveals network clustering based on stimulation.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    _detach_heatmap_interaction(plot)
    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle(
            "Pairwise Pearson Correlation (Zero-Lag - Deconvolved DF/F) "
            "(Sorted - Need ≥2 ROIs)"
        )
        return

    # Get correlation matrix from database
    corr_matrix, roi_labels = _get_dec_dff_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if corr_matrix is None or roi_labels is None:
        plot.setTitle("Deconvolved DF/F Correlation (Sorted - No data)")
        return

    # Reorder matrix according to sorted ROIs
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        corr_matrix, roi_labels, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Deconvolved DF/F Correlation (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap (Pearson correlation ranges from -1 to 1)
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(-1.0, 1.0, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = (
        f"Pairwise Pearson Correlation (Zero-Lag - Deconvolved DF/F) "
        f"(Sorted: {n_stim} Stim, {n_non_stim} Non-Stim)"
    )

    # Add medians
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block (green rectangle)
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)

        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)

    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)


def _plot_sorted_dec_dff_correlation_windowed_by_stim(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    window_ms: float = 250.0,
) -> None:
    """Plot windowed Pearson correlation around stimulation frames.

    Calculates correlation only using time points within ±window_ms of each
    LED pulse, revealing how neurons correlate specifically during stimulation.
    ROIs are sorted by stimulation status (stimulated first, then non-stimulated).

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        FOV name
    rois : list[int] | None
        ROI filter
    run_id : int | None
        Analysis run ID
    window_ms : float
        Time window around each stimulation frame (milliseconds), default 250ms
    """
    from cali.sqlmodel._model import AnalysisSettings, CaliResult, Traces

    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()
    _detach_heatmap_interaction(plot)
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.setTitle("Windowed Correlation (Sorted - Need run ID)")
        return

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle(
            "Windowed Pearson Correlation (Stim Windows) (Sorted - Need ≥2 ROIs)"
        )
        return

    # Get analysis settings to extract LED pulse frames and frame rate
    with Session(engine) as session:
        # Get the CaliResult for this run to find analysis_settings_id
        result = session.exec(select(CaliResult).where(CaliResult.id == run_id)).first()

        if result is None or result.analysis_settings_id is None:
            plot.setTitle("Windowed Correlation (Sorted - No analysis settings)")
            return

        # Get analysis settings
        analysis_settings = session.exec(
            select(AnalysisSettings).where(
                AnalysisSettings.id == result.analysis_settings_id
            )
        ).first()

        if analysis_settings is None:
            plot.setTitle("Windowed Correlation (Sorted - No analysis settings)")
            return

        if analysis_settings.led_pulse_on_frames is None:
            plot.setTitle("Windowed Correlation (Sorted - No LED pulse frames defined)")
            return

        led_pulse_frames = analysis_settings.led_pulse_on_frames
        frame_rate = analysis_settings.frame_rate

        # Convert window_ms to frames
        window_frames = int((window_ms / 1000.0) * frame_rate)

        # Get FOV to access traces
        fov = session.exec(select(FOV).where(FOV.name == fov_name)).first()

        if fov is None:
            plot.setTitle("Windowed Correlation (Sorted - FOV not found)")
            return

        # Load all ROIs with their traces for this analysis run
        roi_data = {}
        for roi in fov.rois:
            if roi.label_value not in all_sorted:
                continue

            # Get the trace for this analysis run
            trace = session.exec(
                select(Traces).where(
                    Traces.roi_id == roi.id,
                    Traces.analysis_result_id == run_id,
                )
            ).first()

            if trace is None or trace.dec_dff is None:
                continue

            roi_data[roi.label_value] = np.array(trace.dec_dff)

    if len(roi_data) < 2:
        plot.setTitle("Windowed Correlation (Sorted - Insufficient trace data)")
        return

    # Extract windowed segments around each LED pulse
    windowed_traces = {}
    for roi_label, full_trace in roi_data.items():
        segments = []
        for pulse_frame in led_pulse_frames:
            start_frame = max(0, pulse_frame - window_frames)
            end_frame = min(len(full_trace), pulse_frame + window_frames + 1)
            segments.append(full_trace[start_frame:end_frame])

        # Concatenate all segments for this ROI
        windowed_traces[roi_label] = np.concatenate(segments)

    # Calculate Pearson correlation on windowed traces
    roi_labels_ordered = sorted(windowed_traces.keys())
    traces_array = np.array([windowed_traces[label] for label in roi_labels_ordered])

    # Compute zero-lag Pearson correlation
    from cali.analysis._util import _compute_zero_lag_corr_matrix

    corr_matrix = _compute_zero_lag_corr_matrix(list(traces_array))

    if corr_matrix is None:
        plot.setTitle("Windowed Correlation (Sorted - Failed to compute)")
        return

    # Reorder matrix according to sorted ROIs (stim first, then non-stim)
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        corr_matrix, roi_labels_ordered, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Windowed Correlation (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap (Pearson correlation ranges from -1 to 1)
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(-1.0, 1.0, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = (
        f"Pairwise Pearson Correlation (Stim Windows ±{int(window_ms)}ms - "
        f"Deconvolved DF/F) (Sorted: {n_stim} Stim, {n_non_stim} Non-Stim)"
    )

    # Add medians
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)

        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)

    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)


def _plot_sorted_dec_dff_correlation_windowed_non_stim(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    window_ms: float = 250.0,
) -> None:
    """Plot windowed Pearson correlation OUTSIDE stimulation frames.

    Calculates correlation only using time points OUTSIDE ±window_ms of each
    LED pulse, revealing how neurons correlate during baseline/recovery periods.
    ROIs are sorted by stimulation status (stimulated first, then non-stimulated).

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        FOV name
    rois : list[int] | None
        ROI filter
    run_id : int | None
        Analysis run ID
    window_ms : float
        Time window around each stimulation frame (milliseconds), default 250ms
    """
    from cali.sqlmodel._model import AnalysisSettings, CaliResult, Traces

    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()
    _detach_heatmap_interaction(plot)
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.setTitle("Non-Stim Windowed Correlation (Sorted - Need run ID)")
        return

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle("Non-Stim Windowed Pearson Correlation (Sorted - Need ≥2 ROIs)")
        return

    # Get analysis settings to extract LED pulse frames and frame rate
    with Session(engine) as session:
        # Get the CaliResult for this run to find analysis_settings_id
        result = session.exec(select(CaliResult).where(CaliResult.id == run_id)).first()

        if result is None or result.analysis_settings_id is None:
            plot.setTitle(
                "Non-Stim Windowed Correlation (Sorted - No analysis settings)"
            )
            return

        # Get analysis settings
        analysis_settings = session.exec(
            select(AnalysisSettings).where(
                AnalysisSettings.id == result.analysis_settings_id
            )
        ).first()

        if analysis_settings is None:
            plot.setTitle(
                "Non-Stim Windowed Correlation (Sorted - No analysis settings)"
            )
            return

        if analysis_settings.led_pulse_on_frames is None:
            plot.setTitle(
                "Non-Stim Windowed Correlation (Sorted - No LED pulse frames defined)"
            )
            return

        led_pulse_frames = sorted(analysis_settings.led_pulse_on_frames)
        frame_rate = analysis_settings.frame_rate

        # Convert window_ms to frames
        window_frames = int((window_ms / 1000.0) * frame_rate)

        # Get FOV to access traces
        fov = session.exec(select(FOV).where(FOV.name == fov_name)).first()

        if fov is None:
            plot.setTitle("Non-Stim Windowed Correlation (Sorted - FOV not found)")
            return

        # Load all ROIs with their traces for this analysis run
        roi_data = {}
        for roi in fov.rois:
            if roi.label_value not in all_sorted:
                continue

            # Get the trace for this analysis run
            trace = session.exec(
                select(Traces).where(
                    Traces.roi_id == roi.id,
                    Traces.analysis_result_id == run_id,
                )
            ).first()

            if trace is None or trace.dec_dff is None:
                continue

            roi_data[roi.label_value] = np.array(trace.dec_dff)

    if len(roi_data) < 2:
        plot.setTitle(
            "Non-Stim Windowed Correlation (Sorted - Insufficient trace data)"
        )
        return

    # Extract NON-windowed segments (everything OUTSIDE ±window_ms of LED pulses)
    windowed_traces = {}
    for roi_label, full_trace in roi_data.items():
        segments = []
        trace_length = len(full_trace)

        # Segment before first pulse
        if led_pulse_frames[0] - window_frames > 0:
            segments.append(full_trace[0 : led_pulse_frames[0] - window_frames])

        # Segments between pulses
        for i in range(len(led_pulse_frames) - 1):
            start_frame = led_pulse_frames[i] + window_frames + 1
            end_frame = led_pulse_frames[i + 1] - window_frames
            if start_frame < end_frame:
                segments.append(full_trace[start_frame:end_frame])

        # Segment after last pulse
        if led_pulse_frames[-1] + window_frames + 1 < trace_length:
            segments.append(full_trace[led_pulse_frames[-1] + window_frames + 1 :])

        # Concatenate all non-stim segments for this ROI
        if segments:
            windowed_traces[roi_label] = np.concatenate(segments)
        else:
            # If no non-stim data available, skip this ROI
            continue

    if len(windowed_traces) < 2:
        plot.setTitle(
            "Non-Stim Windowed Correlation (Sorted - Insufficient non-stim data)"
        )
        return

    # Calculate Pearson correlation on non-stim windowed traces
    roi_labels_ordered = sorted(windowed_traces.keys())
    traces_array = np.array([windowed_traces[label] for label in roi_labels_ordered])

    # Compute zero-lag Pearson correlation
    from cali.analysis._util import _compute_zero_lag_corr_matrix

    corr_matrix = _compute_zero_lag_corr_matrix(list(traces_array))

    if corr_matrix is None:
        plot.setTitle("Non-Stim Windowed Correlation (Sorted - Failed to compute)")
        return

    # Reorder matrix according to sorted ROIs (stim first, then non-stim)
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        corr_matrix, roi_labels_ordered, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Non-Stim Windowed Correlation (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap (Pearson correlation ranges from -1 to 1)
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(-1.0, 1.0, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = (
        f"Pairwise Pearson Correlation (Non-Stim Periods, Excluding "
        f"±{int(window_ms)}ms - Deconvolved DF/F) (Sorted: {n_stim} Stim, "
        f"{n_non_stim} Non-Stim)"
    )

    # Add medians
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)

        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)

    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)


def _plot_sorted_spike_correlation_windowed_by_stim(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    window_ms: float = 250.0,
) -> None:
    """Plot windowed Pearson correlation for inferred spikes around stimulation frames.

    Calculates correlation only using time points within ±window_ms of each
    LED pulse on thresholded inferred spike trains, revealing how spike
    correlations change specifically during stimulation.
    ROIs are sorted by stimulation status (stimulated first, then non-stimulated).

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        FOV name
    rois : list[int] | None
        ROI filter
    run_id : int | None
        Analysis run ID
    window_ms : float
        Time window around each stimulation frame (milliseconds), default 250ms
    """
    from cali.sqlmodel._model import AnalysisSettings, CaliResult, Traces

    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    _detach_heatmap_interaction(plot)
    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.setTitle("Windowed Spike Correlation (Sorted - Need run ID)")
        return

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle(
            "Windowed Spike Correlation (Stim Windows) (Sorted - Need ≥2 ROIs)"
        )
        return

    # Get analysis settings to extract LED pulse frames and frame rate
    with Session(engine) as session:
        # Get the CaliResult for this run to find analysis_settings_id
        result = session.exec(select(CaliResult).where(CaliResult.id == run_id)).first()

        if result is None or result.analysis_settings_id is None:
            plot.setTitle("Windowed Spike Correlation (Sorted - No analysis settings)")
            return

        # Get analysis settings
        analysis_settings = session.exec(
            select(AnalysisSettings).where(
                AnalysisSettings.id == result.analysis_settings_id
            )
        ).first()

        if analysis_settings is None:
            plot.setTitle("Windowed Spike Correlation (Sorted - No analysis settings)")
            return

        if analysis_settings.led_pulse_on_frames is None:
            plot.setTitle(
                "Windowed Spike Correlation (Sorted - No LED pulse frames defined)"
            )
            return

        led_pulse_frames = analysis_settings.led_pulse_on_frames
        frame_rate = analysis_settings.frame_rate

        # Convert window_ms to frames
        window_frames = int((window_ms / 1000.0) * frame_rate)

        # Get FOV to access traces
        fov = session.exec(select(FOV).where(FOV.name == fov_name)).first()

        if fov is None:
            plot.setTitle("Windowed Spike Correlation (Sorted - FOV not found)")
            return

        # Load all ROIs with their spike traces for this analysis run
        roi_data = {}
        for roi in fov.rois:
            if roi.label_value not in all_sorted:
                continue

            # Get the trace and analysis for this analysis run
            trace = session.exec(
                select(Traces).where(
                    Traces.roi_id == roi.id,
                    Traces.analysis_result_id == run_id,
                )
            ).first()

            if trace is None or trace.inferred_spikes is None:
                continue

            # Get the threshold for this ROI
            from cali.sqlmodel._model import DataAnalysis

            data_analysis = session.exec(
                select(DataAnalysis).where(
                    DataAnalysis.roi_id == roi.id,
                    DataAnalysis.analysis_result_id == run_id,
                )
            ).first()

            if data_analysis is None:
                continue

            threshold = data_analysis.inferred_spikes_threshold or 0.0

            # Apply threshold to inferred spikes
            inferred_spikes = np.array(trace.inferred_spikes, dtype=float)
            thresholded_spikes = np.where(
                inferred_spikes > threshold, inferred_spikes, 0.0
            )

            roi_data[roi.label_value] = thresholded_spikes

    if len(roi_data) < 2:
        plot.setTitle("Windowed Spike Correlation (Sorted - Insufficient trace data)")
        return

    # Extract windowed segments around each LED pulse
    windowed_traces = {}
    for roi_label, full_trace in roi_data.items():
        segments = []
        for pulse_frame in led_pulse_frames:
            start_frame = max(0, pulse_frame - window_frames)
            end_frame = min(len(full_trace), pulse_frame + window_frames + 1)
            segments.append(full_trace[start_frame:end_frame])

        # Concatenate all segments for this ROI
        windowed_traces[roi_label] = np.concatenate(segments)

    # Calculate Pearson correlation on windowed spike traces
    roi_labels_ordered = sorted(windowed_traces.keys())
    traces_array = np.array([windowed_traces[label] for label in roi_labels_ordered])

    # Compute zero-lag Pearson correlation
    from cali.analysis._util import _compute_zero_lag_corr_matrix

    corr_matrix = _compute_zero_lag_corr_matrix(list(traces_array))

    if corr_matrix is None:
        plot.setTitle("Windowed Spike Correlation (Sorted - Failed to compute)")
        return

    # Reorder matrix according to sorted ROIs (stim first, then non-stim)
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        corr_matrix, roi_labels_ordered, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Windowed Spike Correlation (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap (Pearson correlation ranges from -1 to 1)
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(-1.0, 1.0, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = (
        f"Pairwise Pearson Correlation (Stim Windows ±{int(window_ms)}ms - "
        f"Inferred Spikes) (Sorted: {n_stim} Stim, {n_non_stim} Non-Stim)"
    )

    # Add medians
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)

        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)

    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)


def _plot_sorted_spike_correlation_windowed_non_stim(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    window_ms: float = 250.0,
) -> None:
    """Plot windowed spike correlation OUTSIDE stimulation frames.

    Calculates correlation only using time points OUTSIDE ±window_ms of each
    LED pulse, revealing how neurons correlate during baseline/recovery periods.
    ROIs are sorted by stimulation status (stimulated first, then non-stimulated).

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        FOV name
    rois : list[int] | None
        ROI filter
    run_id : int | None
        Analysis run ID
    window_ms : float
        Time window around each stimulation frame (milliseconds), default 250ms
    """
    from cali.sqlmodel._model import AnalysisSettings, CaliResult, DataAnalysis, Traces

    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()
    _detach_heatmap_interaction(plot)
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.setTitle("Non-Stim Spike Correlation (Sorted - Need run ID)")
        return

    # Get sorted ROI lists
    all_sorted, stim_rois, non_stim_rois = _get_sorted_rois_by_stimulation(
        engine, fov_name, rois
    )

    if len(all_sorted) < 2:
        plot.setTitle("Non-Stim Spike Correlation (Sorted - Need ≥2 ROIs)")
        return

    # Get analysis settings and spike threshold
    with Session(engine) as session:
        # Get the CaliResult for this run to find analysis_settings_id
        result = session.exec(select(CaliResult).where(CaliResult.id == run_id)).first()

        if result is None or result.analysis_settings_id is None:
            plot.setTitle("Non-Stim Spike Correlation (Sorted - No analysis settings)")
            return

        # Get analysis settings
        analysis_settings = session.exec(
            select(AnalysisSettings).where(
                AnalysisSettings.id == result.analysis_settings_id
            )
        ).first()

        if analysis_settings is None:
            plot.setTitle("Non-Stim Spike Correlation (Sorted - No analysis settings)")
            return

        if analysis_settings.led_pulse_on_frames is None:
            plot.setTitle(
                "Non-Stim Spike Correlation (Sorted - No LED pulse frames defined)"
            )
            return

        led_pulse_frames = sorted(analysis_settings.led_pulse_on_frames)
        frame_rate = analysis_settings.frame_rate

        # Convert window_ms to frames
        window_frames = int((window_ms / 1000.0) * frame_rate)

        # Get FOV to access traces
        fov = session.exec(select(FOV).where(FOV.name == fov_name)).first()

        if fov is None:
            plot.setTitle("Non-Stim Spike Correlation (Sorted - FOV not found)")
            return

        # Load all ROIs with their traces for this analysis run
        roi_data = {}
        for roi in fov.rois:
            if roi.label_value not in all_sorted:
                continue

            # Get the trace for this analysis run
            trace = session.exec(
                select(Traces).where(
                    Traces.roi_id == roi.id,
                    Traces.analysis_result_id == run_id,
                )
            ).first()

            if trace is None or trace.inferred_spikes is None:
                continue

            # Get the threshold for this ROI
            from cali.sqlmodel._model import DataAnalysis

            data_analysis = session.exec(
                select(DataAnalysis).where(
                    DataAnalysis.roi_id == roi.id,
                    DataAnalysis.analysis_result_id == run_id,
                )
            ).first()

            if data_analysis is None:
                continue

            threshold = data_analysis.inferred_spikes_threshold or 0.0

            # Apply threshold to inferred spikes
            inferred_spikes = np.array(trace.inferred_spikes, dtype=float)
            thresholded_spikes = np.where(
                inferred_spikes > threshold, inferred_spikes, 0.0
            )

            roi_data[roi.label_value] = thresholded_spikes

    if len(roi_data) < 2:
        plot.setTitle("Non-Stim Spike Correlation (Sorted - Insufficient trace data)")
        return

    # Extract NON-windowed segments (everything OUTSIDE ±window_ms of LED pulses)
    windowed_traces = {}
    for roi_label, full_trace in roi_data.items():
        segments = []
        trace_length = len(full_trace)

        # Segment before first pulse
        if led_pulse_frames[0] - window_frames > 0:
            segments.append(full_trace[0 : led_pulse_frames[0] - window_frames])

        # Segments between pulses
        for i in range(len(led_pulse_frames) - 1):
            start_frame = led_pulse_frames[i] + window_frames + 1
            end_frame = led_pulse_frames[i + 1] - window_frames
            if start_frame < end_frame:
                segments.append(full_trace[start_frame:end_frame])

        # Segment after last pulse
        if led_pulse_frames[-1] + window_frames + 1 < trace_length:
            segments.append(full_trace[led_pulse_frames[-1] + window_frames + 1 :])

        # Concatenate all non-stim segments for this ROI
        if segments:
            windowed_traces[roi_label] = np.concatenate(segments)
        else:
            # If no non-stim data available, skip this ROI
            continue

    if len(windowed_traces) < 2:
        plot.setTitle(
            "Non-Stim Spike Correlation (Sorted - Insufficient non-stim data)"
        )
        return

    # Calculate Pearson correlation on non-stim windowed traces
    roi_labels_ordered = sorted(windowed_traces.keys())
    traces_array = np.array([windowed_traces[label] for label in roi_labels_ordered])

    # Compute zero-lag Pearson correlation
    from cali.analysis._util import _compute_zero_lag_corr_matrix

    corr_matrix = _compute_zero_lag_corr_matrix(list(traces_array))

    if corr_matrix is None:
        plot.setTitle("Non-Stim Spike Correlation (Sorted - Failed to compute)")
        return

    # Reorder matrix according to sorted ROIs (stim first, then non-stim)
    reordered_matrix, final_rois = _reorder_matrix_by_roi_list(
        corr_matrix, roi_labels_ordered, all_sorted
    )

    if reordered_matrix is None or len(final_rois) < 2:
        plot.setTitle("Non-Stim Spike Correlation (Sorted - Insufficient ROIs)")
        return

    # Plot the heatmap (Pearson correlation ranges from -1 to 1)
    img = pg.ImageItem(reordered_matrix)
    img.setLookupTable(CMAP.getLookupTable(-1.0, 1.0, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb.invertY(True)
    vb.setAspectLocked(True)

    # Build title with counts
    n_stim = len([r for r in final_rois if r in stim_rois])
    n_non_stim = len([r for r in final_rois if r in non_stim_rois])

    # Calculate medians: stimulated block, non-stimulated block, and global
    mask = ~np.eye(reordered_matrix.shape[0], dtype=bool)
    global_median = np.median(reordered_matrix[mask])

    # Stimulated block (top-left n_stim x n_stim)
    if n_stim > 1:
        stim_block = reordered_matrix[:n_stim, :n_stim]
        stim_mask = ~np.eye(n_stim, dtype=bool)
        stim_median = np.median(stim_block[stim_mask])
    else:
        stim_median = np.nan

    # Non-stimulated block (bottom-right n_non_stim x n_non_stim)
    if n_non_stim > 1:
        non_stim_block = reordered_matrix[n_stim:, n_stim:]
        non_stim_mask = ~np.eye(n_non_stim, dtype=bool)
        non_stim_median = np.median(non_stim_block[non_stim_mask])
    else:
        non_stim_median = np.nan

    title = (
        f"Pairwise Pearson Correlation (Non-Stim Periods, Excluding "
        f"±{int(window_ms)}ms - Inferred Spikes) (Sorted: {n_stim} Stim, "
        f"{n_non_stim} Non-Stim)"
    )

    # Add medians
    if not np.isnan(stim_median):
        title += f" | Stim median: {stim_median:.3f}"
    if not np.isnan(non_stim_median):
        title += f" | Non-stim median: {non_stim_median:.3f}"
    title += f" | Global median: {global_median:.3f}"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CMAP_NAME
    )

    # Add visual marker for stimulated ROI block
    if n_stim > 0:
        rect = pg.QtWidgets.QGraphicsRectItem(0, 0, n_stim, n_stim)
        rect.setPen(pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH))
        rect.setBrush(pg.mkBrush(None))
        plot.addItem(rect)

        # Add legend
        if hasattr(widget, "legend") and widget.legend is not None:
            widget.legend.clear()
            widget.legend.addItem(
                pg.PlotDataItem(
                    pen=pg.mkPen(color=STIM_RECTANGLE_COLOR, width=STIM_RECTANGLE_WIDTH)
                ),
                "Stimulated ROIs",
            )
            widget.legend.setVisible(True)

    # Add hover + click interaction
    _attach_heatmap_interaction(widget, plot, title, vb, final_rois, reordered_matrix)
