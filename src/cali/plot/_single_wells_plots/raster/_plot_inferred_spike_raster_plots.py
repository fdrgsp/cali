from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


def _generate_spike_raster_plot(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
    amplitude_colors: bool = False,
    colorbar: bool = False,
) -> None:
    """Generate a spike raster plot using thresholded spike data (pyqtgraph)."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setAspectLocked(False)

    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    plot.setTitle("Inferred Spikes Raster Plot")

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

    min_amp = float("inf")
    max_amp = float("-inf")

    for roi, traces, data_analysis in roi_data:
        if data_analysis is None or not traces.inferred_spikes:
            continue

        threshold = data_analysis.inferred_spikes_threshold or 0.0
        inferred = np.asarray(traces.inferred_spikes, dtype=float)

        # Thresholded spikes
        above_the = inferred > threshold
        if not np.any(above_the):
            continue

        spike_times = np.where(above_the)[0]
        spike_amplitudes = inferred[above_the]

        if spike_times.size == 0:
            continue

        active_rois.append(roi.label_value)
        event_data.append(spike_times.astype(float))
        per_roi_spike_amplitudes.append(spike_amplitudes.astype(float))

        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        if sample_trace is None and traces.corrected_trace is not None:
            sample_trace = traces.corrected_trace

        if amplitude_colors and spike_amplitudes.size > 0:
            min_amp = min(min_amp, float(spike_amplitudes.min()))
            max_amp = max(max_amp, float(spike_amplitudes.max()))

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

    if amplitude_colors and np.isfinite(min_amp) and np.isfinite(max_amp):
        vmin, vmax = _compute_amp_norm_bounds(min_amp, max_amp)
        norm_amp_color = Normalize(vmin=vmin, vmax=vmax)
        cmap = colormaps.get_cmap("viridis")

        for amps in per_roi_spike_amplitudes:
            row_cols: list[tuple[int, int, int, int]] = []
            for a in amps:
                rgba = cmap(norm_amp_color(float(a)))  # floats in [0,1]
                r, g, b, a_ = [int(255 * c) for c in rgba]
                row_cols.append((r, g, b, a_))
            per_roi_colors.append(row_cols)
    else:
        amplitude_colors = False
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
                }
            )

        item = pg.ScatterPlotItem(spots=spots)
        item.setProperty("roi_label", str(active_rois[row_idx]))
        plot.addItem(item)

    # ------------------------ Axes ------------------------ #
    plot.setLabel("left", "ROI (rows)")
    _update_time_axis_pg_frames(plot, rois_rec_time, sample_trace)

    # Hide y tick values
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    plot.getViewBox().enableAutoRange(x=True, y=True)

    # ------------------------ Colorbar ------------------------ #
    if colorbar and amplitude_colors and np.isfinite(min_amp) and np.isfinite(max_amp):
        _add_colorbar_to_widget(widget, vmin, vmax)

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


def _add_colorbar_to_widget(
    widget: _SingleWellGraphWidget,
    vmin: float,
    vmax: float,
) -> None:
    """Add a ColorBarItem to the widget layout."""
    # Create ColorBarItem
    widget.colorbar = pg.ColorBarItem(
        values=(vmin, vmax),
        colorMap=pg.colormap.get("viridis"),
        width=15,
        label="Spike Amplitude",
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)


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
        idx = round(y)
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
