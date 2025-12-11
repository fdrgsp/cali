from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


def _generate_spike_raster_plot_raw(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Generate a spike raster plot using raw (unthresholded) spike data (pyqtgraph)."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    # Reset ViewBox settings that might have been set by previous plots
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.invertY(True)  # Invert Y so row 0 (ROI 1) appears at BOTTOM visually

    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    plot.setTitle("Inferred Spike Events (binary) Raster Plot (Raw)")

    # ------------------------ Query DB ------------------------ #
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data: list[tuple[ROI, Traces, DataAnalysis | None]] = session.exec(
            stmt
        ).all()

    if not roi_data:
        cali_logger.warning("No ROI data found for raw spike raster plot.")
        _draw_centered_message_pg(
            plot,
            "No ROI data found for this FOV.\nPlease check the selected run and FOV.",
        )
        return

    # ------------------------ Collect events ------------------------ #
    event_data: list[np.ndarray] = []  # per-ROI array of spike indices
    per_roi_spike_amplitudes: list[np.ndarray] = []
    rois_rec_time: list[float] = []
    active_rois: list[int] = []
    sample_trace: list[float] | None = None

    for roi, traces, data_analysis in roi_data:
        if traces is None or not traces.inferred_spikes:
            continue

        inferred = np.asarray(traces.inferred_spikes, dtype=float)

        # Raw spikes: detect any positive values (no thresholding)
        positive_vals = inferred > 0
        if not np.any(positive_vals):
            continue

        # rising edges: 0 -> positive transitions
        rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
        spike_times = np.where(rising)[0]
        spike_amplitudes = inferred[spike_times]

        if spike_times.size == 0:
            continue

        active_rois.append(roi.label_value)
        event_data.append(spike_times.astype(float))
        per_roi_spike_amplitudes.append(spike_amplitudes.astype(float))

        if data_analysis and data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        if sample_trace is None and traces.inferred_spikes is not None:
            sample_trace = traces.inferred_spikes

    if not event_data:
        cali_logger.warning("No raw spike data found for the selected ROIs and run.")
        _draw_centered_message_pg(
            plot,
            "No spike data found.\nPlease check the selected run and FOV.",
        )
        return

    # ------------------------ Colors per spike ------------------------ #
    per_roi_colors: list[list[tuple[int, int, int, int]]] = []

    for times in event_data:
        # same length, all white
        per_roi_colors.append([(255, 255, 255, 255)] * len(times))

    # ------------------------ Plot raster (one row per ROI) ------------------------ #
    for row_idx, (times, row_colors) in enumerate(zip(event_data, per_roi_colors)):
        if times.size == 0:
            continue

        y_vals = np.full_like(times, row_idx, dtype=float)
        spots = []
        for x, y, color in zip(times, y_vals, row_colors):
            spots.append(
                {
                    "pos": (float(x), float(y)),
                    "brush": pg.mkBrush(*color),
                    "pen": None,
                    "size": 3,
                    "symbol": "s",
                }
            )

        item = pg.ScatterPlotItem(spots=spots)
        item.setProperty("roi_label", str(active_rois[row_idx]))
        plot.addItem(item)

    # ------------------------ Axes ------------------------ #
    plot.setLabel("left", "ROI")
    _update_time_axis_pg_frames(plot, rois_rec_time, sample_trace)

    # Hide y tick values
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)
    y_axis.enableAutoSIPrefix(False)

    plot.getViewBox().enableAutoRange(x=True, y=True)

    # ------------------------ Click → roiSelected ------------------------ #
    _attach_click_handlers_raster(widget, plot, active_rois)


def _generate_spike_raster_plot(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Generate a spike raster plot using thresholded spike data (pyqtgraph)."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    # Reset ViewBox settings that might have been set by previous plots
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.invertY(True)  # Invert Y so row 0 (ROI 1) appears at BOTTOM visually

    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    plot.setTitle("Inferred Spike Events (binary) Raster Plot (Thresholded)")

    # ------------------------ Query DB ------------------------ #
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
        )

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data: list[tuple[ROI, Traces, DataAnalysis]] = session.exec(stmt).all()

    if not roi_data:
        cali_logger.warning("No ROI data found for spike raster plot.")
        _draw_centered_message_pg(
            plot,
            "No ROI data found for this FOV.\nPlease check the selected run and FOV.",
        )
        return

    # ------------------------ Collect events ------------------------ #
    event_data: list[np.ndarray] = []  # per-ROI array of spike indices
    per_roi_spike_amplitudes: list[np.ndarray] = []
    rois_rec_time: list[float] = []
    active_rois: list[int] = []
    sample_trace: list[float] | None = None

    float("inf")
    float("-inf")

    for roi, traces, data_analysis in roi_data:
        if data_analysis is None or not traces.inferred_spikes:
            continue

        threshold = data_analysis.inferred_spikes_threshold or 0.0
        inferred = np.asarray(traces.inferred_spikes, dtype=float)

        # Thresholded spikes
        above_the = inferred > threshold
        if not np.any(above_the):
            continue

        # rising edges: 0 -> 1 transitions
        rising = above_the & ~np.concatenate(([False], above_the[:-1]))
        spike_times = np.where(rising)[0]
        spike_amplitudes = inferred[spike_times]

        spike_amplitudes = inferred[above_the]

        if spike_times.size == 0:
            continue

        active_rois.append(roi.label_value)
        event_data.append(spike_times.astype(float))
        per_roi_spike_amplitudes.append(spike_amplitudes.astype(float))

        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        if sample_trace is None and traces.inferred_spikes is not None:
            sample_trace = traces.inferred_spikes

    if not event_data:
        cali_logger.warning(
            "No spike data above threshold for the selected ROIs and run."
        )
        _draw_centered_message_pg(
            plot,
            "No spike data above threshold.\n"
            "Try adjusting thresholds or selecting different ROIs.",
        )
        return

    # ------------------------ Colors per spike ------------------------ #
    per_roi_colors: list[list[tuple[int, int, int, int]]] = []

    for times in event_data:
        # same length, all white
        per_roi_colors.append([(255, 255, 255, 255)] * len(times))

    # ------------------------ Plot raster (one row per ROI) ------------------------ #
    for row_idx, (times, row_colors) in enumerate(zip(event_data, per_roi_colors)):
        if times.size == 0:
            continue

        y_vals = np.full_like(times, row_idx, dtype=float)
        spots = []
        for x, y, color in zip(times, y_vals, row_colors):
            spots.append(
                {
                    "pos": (float(x), float(y)),
                    "brush": pg.mkBrush(*color),
                    "pen": None,
                    "size": 3,
                    "symbol": "s",
                }
            )

        item = pg.ScatterPlotItem(spots=spots)
        item.setProperty("roi_label", str(active_rois[row_idx]))
        plot.addItem(item)

    # ------------------------ Axes ------------------------ #
    plot.setLabel("left", "ROI")
    _update_time_axis_pg_frames(plot, rois_rec_time, sample_trace)

    # Hide y tick values
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)
    y_axis.enableAutoSIPrefix(False)

    plot.getViewBox().enableAutoRange(x=True, y=True)

    # ------------------------ Click → roiSelected ------------------------ #
    _attach_click_handlers_raster(widget, plot, active_rois)


def _draw_centered_message_pg(plot: pg.PlotItem, text: str) -> None:
    """Utility to draw a centered multi-line message and hide axes (pyqtgraph)."""
    plot.clear()
    text_item = pg.TextItem(text, anchor=(0.5, 0.5), color="w")
    plot.addItem(text_item)
    text_item.setPos(0, 0)
    plot.getViewBox().autoRange()
    plot.setTitle("")  # leave title empty for message


def _compute_amp_norm_bounds(min_amp: float, max_amp: float) -> tuple[float, float]:
    """Compute robust vmin/vmax for amplitude normalization (same logic as MPL)."""
    if not np.isfinite(min_amp) or not np.isfinite(max_amp):
        return 0.0, 1.0

    # Use a reduced range (e.g. 60% of full) to make mid/high values more visible
    vmax = min_amp + (max_amp - min_amp) * 0.6
    if vmax <= min_amp:
        vmax = max_amp
    if vmax <= min_amp:
        vmax = min_amp + 0.1  # tiny positive range if everything is almost equal

    vmin = min_amp
    return vmin, vmax


def _update_time_axis_pg_frames(
    plot: pg.PlotItem,
    rois_rec_time: list[float],
    trace: list[float] | None,
) -> None:
    """Set bottom axis as time (s) if total recording time is available, else frames."""
    if trace is None or not rois_rec_time or sum(rois_rec_time) <= 0:
        plot.setLabel("bottom", "Frames")
        return

    total_frames = len(trace) if trace is not None else 1
    if total_frames <= 1:
        plot.setLabel("bottom", "Frames")
        return

    avg_rec_time = int(np.mean(rois_rec_time))
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    tick_interval = avg_rec_time / total_frames
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]

    axis = plot.getAxis("bottom")
    axis.setTicks([list(zip(x_ticks.tolist(), x_labels))])
    plot.setLabel("bottom", "Time (s)")


def _attach_click_handlers_raster(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    active_roi_labels: list[int],
) -> None:
    """Map clicked Y row → ROI label and emit widget.roiSelected."""
    from pyqtgraph import Point

    scene = plot.scene()
    vb = plot.getViewBox()

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        p: Point = vb.mapSceneToView(pos)
        y = float(p.y())
        # With invertY(True), y increases downward; floor gives correct row
        idx = int(np.floor(y))
        if 0 <= idx < len(active_roi_labels):
            widget.roiSelected.emit(str(active_roi_labels[idx]))

    old_click = plot.property("spike_raster_click_handler")
    if old_click is not None:
        try:
            scene.sigMouseClicked.disconnect(old_click)
        except (TypeError, RuntimeError):
            pass

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("spike_raster_click_handler", _on_mouse_clicked)


def _generate_spike_intensity_heatmap(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Generate intensity heatmap with spike data color-coded.

    Each ROI is displayed as a horizontal row, with the full inferred spike
    signal represented by color intensity (viridis colormap).
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    # Reset ViewBox settings that might have been set by previous plots
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.invertY(True)  # Reset to default (True = y-axis inverted)

    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    plot.setTitle("Inferred Spikes Heatmap (Raw Signal)")

    # ------------------------ Query DB ------------------------ #
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data: list[tuple[ROI, Traces, DataAnalysis | None]] = session.exec(
            stmt
        ).all()

    if not roi_data:
        cali_logger.warning("No ROI data found for spike intensity heatmap.")
        _draw_centered_message_pg(
            plot,
            "No ROI data found for this FOV.\nPlease check the selected run and FOV.",
        )
        return

    # ------------------------ Collect spike trace data ------------------------ #
    traces_list: list[np.ndarray] = []
    active_rois: list[int] = []
    rois_rec_time: list[float] = []

    for roi, traces, data_analysis in roi_data:
        if traces is None or traces.inferred_spikes is None:
            continue

        spike_trace = np.asarray(traces.inferred_spikes, dtype=float)
        if spike_trace.size == 0:
            continue

        traces_list.append(spike_trace)
        active_rois.append(roi.label_value)

        if data_analysis and data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not traces_list:
        cali_logger.warning("No spike trace data found for intensity heatmap.")
        _draw_centered_message_pg(
            plot,
            "No spike trace data available for this FOV.\n"
            "Please check the selected run.",
        )
        return

    # Stack traces into 2D array (n_rois x n_frames)
    traces_array = np.vstack(traces_list)
    n_rois, n_frames = traces_array.shape

    # Percentile-based bounds (robust to outliers) in raw units
    vmin_raw = float(np.percentile(traces_array, 5))
    vmax_raw = float(np.percentile(traces_array, 95))
    if vmax_raw <= vmin_raw:
        vmax_raw = vmin_raw + 0.1

    # ------------------------ Create heatmap ------------------------ #
    img = pg.ImageItem(traces_array)

    # Treat axis 0 as rows (ROI), axis 1 as columns (frames), no smoothing/downsampling
    img.setOpts(
        axisOrder="row-major",
        autoDownsample=False,
        smooth=False,
        levels=(vmin_raw, vmax_raw),
    )

    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

    plot.addItem(img)

    # Viewbox settings: one flat band per ROI
    vb.invertY(False)
    vb.setLimits(xMin=0, xMax=n_frames, yMin=0, yMax=n_rois)
    vb.setRange(xRange=(0, n_frames), yRange=(0, n_rois))
    vb.enableAutoRange(x=True, y=True)

    # ------------------------ Axes ------------------------ #
    plot.setLabel("left", "ROI")
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)
    y_axis.enableAutoSIPrefix(False)

    # Time axis (using first trace as reference)
    sample_trace = traces_list[0]
    _update_time_axis_pg_frames(plot, rois_rec_time, sample_trace)

    # ------------------------ Colorbar ------------------------ #
    _add_spike_intensity_colorbar_to_widget(widget, vmin_raw, vmax_raw)

    # ------------------------ Click → roiSelected ------------------------ #
    _attach_click_handlers_spike_intensity(widget, plot, active_rois)


def _add_spike_intensity_colorbar_to_widget(
    widget: _SingleWellGraphWidget,
    vmin: float,
    vmax: float,
) -> None:
    """Add a ColorBarItem to the spike intensity heatmap widget layout."""
    # Create ColorBarItem with fixed range (non-interactive)
    widget.colorbar = pg.ColorBarItem(
        values=(vmin, vmax),
        colorMap=pg.colormap.get("viridis"),
        width=15,
        label="Inferred spikes (a.u.)",
        interactive=False,
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)


def _attach_click_handlers_spike_intensity(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    active_roi_labels: list[int],
) -> None:
    """Map clicked Y row → ROI label, and emit roiSelected."""
    from pyqtgraph import Point

    scene = plot.scene()
    vb = plot.getViewBox()

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return

        p: Point = vb.mapSceneToView(pos)
        y = float(p.y())
        # Since invertY(False), y=0 is at bottom, so use floor for correct row
        idx = int(np.floor(y))
        if 0 <= idx < len(active_roi_labels):
            widget.roiSelected.emit(str(active_roi_labels[idx]))

    old_click = plot.property("spike_intensity_heatmap_click_handler")
    if old_click is not None:
        try:
            scene.sigMouseClicked.disconnect(old_click)
        except (TypeError, RuntimeError):
            pass

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("spike_intensity_heatmap_click_handler", _on_mouse_clicked)


def _generate_spike_intensity_heatmap_thresholded(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Generate intensity heatmap with thresholded spike data.

    Each ROI is displayed as a horizontal row, showing only spike events
    that exceed the detection threshold (binary: 0 or spike amplitude).
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    # Reset ViewBox settings that might have been set by previous plots
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.invertY(True)  # Reset to default (True = y-axis inverted)

    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    plot.setTitle("Inferred Spikes Heatmap (Thresholded)")

    # ------------------------ Query DB ------------------------ #
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
        )

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data: list[tuple[ROI, Traces, DataAnalysis]] = session.exec(stmt).all()

    if not roi_data:
        cali_logger.warning(
            "No ROI data found for thresholded spike intensity heatmap."
        )
        _draw_centered_message_pg(
            plot,
            "No ROI data found for this FOV.\nPlease check the selected run and FOV.",
        )
        return

    # ------------------------ Collect thresholded spike data ------------------------ #
    traces_list: list[np.ndarray] = []
    active_rois: list[int] = []
    rois_rec_time: list[float] = []

    for roi, traces, data_analysis in roi_data:
        if traces is None or traces.inferred_spikes is None:
            continue

        threshold = data_analysis.inferred_spikes_threshold or 0.0
        spike_signal = np.asarray(traces.inferred_spikes, dtype=float)

        # Apply threshold: keep amplitudes above threshold, set rest to 0
        thresholded_signal = np.where(spike_signal > threshold, spike_signal, 0.0)

        if thresholded_signal.size == 0 or np.all(thresholded_signal == 0):
            continue

        traces_list.append(thresholded_signal)
        active_rois.append(roi.label_value)

        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not traces_list:
        cali_logger.warning("No thresholded spike data found for intensity heatmap.")
        _draw_centered_message_pg(
            plot,
            "No spike data above threshold for this FOV.\n"
            "Try adjusting thresholds or selecting different ROIs.",
        )
        return

    # Stack traces into 2D array (n_rois x n_frames)
    traces_array = np.vstack(traces_list)
    n_rois, n_frames = traces_array.shape

    # Get non-zero values for robust scaling
    non_zero_vals = traces_array[traces_array > 0]
    if non_zero_vals.size > 0:
        vmin_raw = 0.0  # Always start at 0 for thresholded data
        vmax_raw = float(np.percentile(non_zero_vals, 95))
        if vmax_raw <= vmin_raw:
            vmax_raw = float(non_zero_vals.max())
        if vmax_raw <= vmin_raw:
            vmax_raw = 1.0
    else:
        vmin_raw = 0.0
        vmax_raw = 1.0

    # ------------------------ Create heatmap ------------------------ #
    img = pg.ImageItem(traces_array)

    # Treat axis 0 as rows (ROI), axis 1 as columns (frames)
    img.setOpts(
        axisOrder="row-major",
        autoDownsample=False,
        smooth=False,
        interpolate=False,  # No interpolation for discrete spikes
        levels=(vmin_raw, vmax_raw),
    )

    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

    plot.addItem(img)

    # Viewbox settings: one flat band per ROI
    vb.invertY(False)
    vb.setLimits(xMin=0, xMax=n_frames, yMin=0, yMax=n_rois)
    vb.setRange(xRange=(0, n_frames), yRange=(0, n_rois))
    vb.enableAutoRange(x=True, y=True)

    # ------------------------ Axes ------------------------ #
    plot.setLabel("left", "ROI")
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)
    y_axis.enableAutoSIPrefix(False)

    # Time axis (using first trace as reference)
    sample_trace = traces_list[0]
    _update_time_axis_pg_frames(plot, rois_rec_time, sample_trace)

    # ------------------------ Colorbar ------------------------ #
    _add_spike_intensity_colorbar_to_widget(widget, vmin_raw, vmax_raw)

    # ------------------------ Click → roiSelected ------------------------ #
    _attach_click_handlers_spike_intensity(widget, plot, active_rois)
