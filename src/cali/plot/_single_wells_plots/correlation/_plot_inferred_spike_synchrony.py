from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, cast

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import (
    _get_spike_synchrony,
    _get_spike_synchrony_matrix,
)
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Main plotting entry point (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_spike_synchrony_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot spike-based synchrony analysis (pyqtgraph heatmap)."""
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()

    # Hide shared legend if present (we don't want it here)
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # 1) Get spike trains per ROI
    spike_trains = _get_spike_trains_from_rois(engine, fov_name, rois, run_id)
    if spike_trains is None or len(spike_trains) < 2:
        cali_logger.warning(
            "Insufficient spike data for synchrony analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        plot.setTitle("Spike Synchrony\n(No data)")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # 2) Get lag from analysis settings
    lag = _get_lag(engine, fov_name, rois, run_id)
    if lag is None:
        cali_logger.warning("No valid lag value found for synchrony analysis.")
        plot.setTitle("Spike Synchrony\n(No lag setting)")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # 3) Convert spike trains to spike data dict for synchrony computation
    spike_data_dict = {
        roi_name: cast("list[float]", spike_train.astype(float).tolist())
        for roi_name, spike_train in spike_trains.items()
    }

    # 4) Compute synchrony matrix using cross-correlation method
    synchrony_matrix = _get_spike_synchrony_matrix(
        spike_data_dict,
        method="cross_correlation",
        max_lag=lag,
    )

    if synchrony_matrix is None:
        cali_logger.warning(
            "Failed to compute synchrony matrix. "
            "Ensure spike data is valid and contains sufficient ROIs."
        )
        plot.setTitle("Spike Synchrony\n(Computation failed)")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # 5) Global synchrony metric
    global_synchrony = _get_spike_synchrony(synchrony_matrix) or 0.0
    title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Thresholded Spike Data - Cross-Correlation Method)\n"
    )

    sync = synchrony_matrix
    sync.shape[0]

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
    plot.setLabel("bottom", "ROI index")
    plot.setLabel("left", "ROI index")

    # Hide axis tick labels (to match MPL style)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Synchrony")

    # ROI ordering is the dict key order
    active_roi_ids = [int(roi_id) for roi_id in spike_trains.keys()]

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
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)


# -----------------------------------------------------------------------------#
# Lag retrieval
# -----------------------------------------------------------------------------#
def _get_lag(
    engine: Engine,
    fov_name: str,  # kept for API symmetry; not used directly here
    rois: list[int] | None = None,  # kept for API symmetry; not used directly here
    run_id: int | None = None,
) -> int | None:
    """Get the lag value for synchrony from AnalysisSettings."""
    from cali.sqlmodel._model import (
        AnalysisSettings,
        CaliResult,
        Experiment,
        Plate,
        Well,
    )

    with Session(engine) as session:
        # Get CaliResult and AnalysisSettings for this run
        # via FOV -> Well -> Plate -> Experiment
        stmt = (
            select(CaliResult, AnalysisSettings)
            .join(Experiment, CaliResult.experiment == Experiment.id)
            .join(Plate, Experiment.id == Plate.experiment_id)
            .join(Well, Plate.id == Well.plate_id)
            .join(FOV, Well.id == FOV.well_id)
            .outerjoin(
                AnalysisSettings,
                CaliResult.analysis_settings_id == AnalysisSettings.id,
            )
            .where(col(FOV.name) == fov_name)
        )
        if run_id is not None:
            stmt = stmt.where(col(CaliResult.id) == run_id)

        result_tuple = session.exec(stmt).first()

        if result_tuple is None:
            cali_logger.warning("No analysis settings found for synchrony analysis.")
            return None

        _, analysis_settings = result_tuple
        if analysis_settings is None:
            cali_logger.warning("No analysis settings found for synchrony analysis.")
            return None

        lag = analysis_settings.spikes_sync_cross_corr_lag
        return lag if lag is not None else 5  # Default fallback


# -----------------------------------------------------------------------------#
# Spike train extraction
# -----------------------------------------------------------------------------#
def _get_spike_trains_from_rois(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> dict[str, np.ndarray] | None:
    """Extract spike trains from ROI data.

    Returns
    -------
    dict[str, np.ndarray] | None
        Dictionary mapping ROI label_value strings to binary spike arrays.
    """
    from cali.sqlmodel._model import ROI

    spike_trains: dict[str, np.ndarray] = {}

    # Query ROIs from database with optimized joins
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )

        # IMPORTANT: subset by label_value (to match other plots)
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_results: list[tuple[ROI, Traces, DataAnalysis]] = session.exec(stmt).all()

    if len(roi_results) < 2:
        return None

    for roi, traces, data_analysis in roi_results:
        if traces is None or data_analysis is None:
            continue

        inferred_spikes = traces.inferred_spikes
        inferred_spikes_threshold = data_analysis.inferred_spikes_threshold

        if inferred_spikes is None or inferred_spikes_threshold is None:
            continue

        spikes = np.asarray(inferred_spikes, dtype=float)
        the = float(inferred_spikes_threshold)

        # Threshold and binarize (vectorized)
        spikes[spikes <= the] = 0.0
        spike_train = (spikes > 0.0).astype(bool)

        if spike_train.sum() > 0 and roi.label_value is not None:
            spike_trains[str(roi.label_value)] = spike_train

    return spike_trains if len(spike_trains) >= 2 else None
