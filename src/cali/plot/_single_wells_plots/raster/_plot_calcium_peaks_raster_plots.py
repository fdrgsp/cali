from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


def _generate_raster_plot(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
    amplitude_colors: bool = False,
    colorbar: bool = False,
) -> None:
    """Generate a raster plot using pyqtgraph by querying database directly."""
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

    title = (
        "Calcium Peaks Raster Plot Colored by Amplitude"
        if amplitude_colors
        else "Calcium Peaks Raster Plot"
    )
    plot.setTitle(title)

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
        plot.setTitle("Calcium Peaks Raster Plot\nNo ROI data found for this FOV.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI (rows)")
        return

    # ------------------------ Collect events & metadata ------------------------ #
    event_data: list[np.ndarray] = []
    rois_rec_time: list[float] = []
    active_rois: list[int] = []
    sample_trace: list[float] | None = None

    # We need a version of roi_data filtered for ROIs that have peaks, to keep
    # colors aligned with event_data rows.
    filtered_roi_data: list[tuple[ROI, Traces, DataAnalysis]] = []

    for roi, traces, data_analysis in roi_data:
        if (
            not data_analysis.peaks_dec_dff
            or not data_analysis.peaks_amplitudes_dec_dff
        ):
            continue

        filtered_roi_data.append((roi, traces, data_analysis))

        # Active ROIs in raster order
        active_rois.append(roi.label_value)

        # Recording time (for time axis)
        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        # Use first valid trace as sample for frame count
        if sample_trace is None and traces.corrected_trace is not None:
            sample_trace = traces.corrected_trace

        # Peaks indices per ROI (frames)
        event_data.append(np.asarray(data_analysis.peaks_dec_dff, dtype=float))

    if not event_data:
        plot.setTitle("Calcium Peaks Raster Plot\nNo peak data available for this FOV.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI (rows)")
        return

    # ------------------------ Colors per ROI ------------------------ #
    colors: list[tuple[int, int, int, int]] = []

    if amplitude_colors and filtered_roi_data:
        # Optimized: Concatenate all amplitudes, compute percentiles once
        all_amps = np.concatenate(
            [
                np.asarray(da.peaks_amplitudes_dec_dff, dtype=float)
                for _, _, da in filtered_roi_data
                if da.peaks_amplitudes_dec_dff
            ]
        )

        if all_amps.size > 0:
            # Use percentiles instead of min/max (robust to outliers)
            vmin = float(np.percentile(all_amps, 5))
            vmax = float(np.percentile(all_amps, 95))

            # Ensure valid bounds
            if vmax <= vmin:
                vmax = vmin + 0.1

            cmap = colormaps.get("viridis")
            norm = Normalize(vmin=vmin, vmax=vmax)

            for _roi, _traces, da in filtered_roi_data:
                amps = np.asarray(da.peaks_amplitudes_dec_dff, dtype=float)
                if amps.size == 0:
                    colors.append((255, 255, 255, 255))
                    continue
                avg_amp = float(amps.mean())
                rgba = cmap(norm(avg_amp))  # floats in [0, 1]
                r, g, b, a = [int(255 * c) for c in rgba]
                colors.append((r, g, b, a))
        else:
            # No amplitude data
            amplitude_colors = False
            colors = [(255, 255, 255, 255)] * len(event_data)
    else:
        # Fallback: all white points
        amplitude_colors = False
        colors = [(255, 255, 255, 255)] * len(event_data)

    # ------------------------ Plot raster (one row per ROI) ------------------------ #
    for row_idx, (events, color) in enumerate(zip(event_data, colors)):
        if events.size == 0:
            continue
        y_vals = np.full_like(events, row_idx, dtype=float)
        item = pg.ScatterPlotItem(
            x=events,
            y=y_vals,
            pen=None,
            brush=pg.mkBrush(*color),
            size=3,
        )
        plot.addItem(item)

    # ------------------------ Axes ------------------------ #
    plot.setLabel("left", "ROI (rows)")
    _update_time_axis_pg_frames(plot, rois_rec_time, sample_trace)

    # Hide numeric Y tick labels (rows are just ordinal)
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    plot.getViewBox().enableAutoRange(x=True, y=True)

    # ------------------------ Colorbar ------------------------ #
    if colorbar and amplitude_colors:
        _add_colorbar_to_widget(widget, vmin, vmax)

    # ------------------------ Click → roiSelected ------------------------ #
    _attach_click_handlers_raster(widget, plot, active_rois)


def _update_time_axis_pg_frames(
    plot: pg.PlotItem,
    rois_rec_time: list[float],
    trace: list[float] | None,
) -> None:
    """Set x-axis as time (s) if total recording time is available, else frames."""
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
        label="Amplitude (dec ΔF/F)",
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)


def _attach_click_handlers_raster(
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
        idx = round(y)
        if 0 <= idx < len(active_roi_labels):
            widget.roiSelected.emit(str(active_roi_labels[idx]))

    old_click = plot.property("amp_raster_click_handler")
    if old_click is not None:
        try:
            scene.sigMouseClicked.disconnect(old_click)
        except (TypeError, RuntimeError):
            pass

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("amp_raster_click_handler", _on_mouse_clicked)


def _generate_intensity_heatmap(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Generate an intensity heatmap showing full trace data color-coded by intensity.

    Each ROI is displayed as a horizontal row, with the full trace signal
    represented by color intensity (viridis colormap).
    """
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

    plot.setTitle("Calcium Intensity Heatmap (Deconvolved ΔF/F)")

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
        plot.setTitle("Calcium Intensity Heatmap\nNo ROI data found for this FOV.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI")
        return

    # ------------------------ Collect trace data ------------------------ #
    traces_list: list[np.ndarray] = []
    active_rois: list[int] = []
    rois_rec_time: list[float] = []

    for roi, traces, data_analysis in roi_data:
        if traces is None or traces.dec_dff is None:
            continue

        trace = np.asarray(traces.dec_dff, dtype=float)
        if trace.size == 0:
            continue

        traces_list.append(trace)
        active_rois.append(roi.label_value)

        if data_analysis and data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not traces_list:
        plot.setTitle(
            "Calcium Intensity Heatmap\nNo trace data available for this FOV."
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI")
        return

    # Stack traces into 2D array (n_rois x n_frames)
    traces_array = np.vstack(traces_list)
    _n_rois, _n_frames = traces_array.shape

    # Compute percentile-based bounds (robust to outliers)
    vmin_raw = float(np.percentile(traces_array, 5))
    vmax_raw = float(np.percentile(traces_array, 95))

    # Ensure valid bounds
    if vmax_raw <= vmin_raw:
        vmax_raw = vmin_raw + 0.1

    # Normalize to [0, 1] for proper colormap visualization
    traces_normalized = (traces_array - vmin_raw) / (vmax_raw - vmin_raw)
    traces_normalized = np.clip(traces_normalized, 0, 1)

    # ------------------------ Create heatmap ------------------------ #
    img = pg.ImageItem(traces_normalized)

    # Apply viridis colormap (data is now in [0, 1])
    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0, 1, 256))
    img.setLevels((0, 1))

    plot.addItem(img)

    # Viewbox settings
    vb.invertY(False)  # Keep top ROI at top
    vb.setAspectLocked(False)  # Allow independent x/y scaling
    vb.enableAutoRange(x=True, y=True)

    # ------------------------ Axes ------------------------ #
    plot.setLabel("left", "ROI")

    # Hide Y tick values (ROI indices are just ordinal)
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    # Time axis (using first trace as reference)
    sample_trace = traces_list[0] if traces_list else None
    _update_time_axis_pg_frames(plot, rois_rec_time, sample_trace)

    # ------------------------ Colorbar ------------------------ #
    _add_intensity_colorbar_to_widget(widget, vmin_raw, vmax_raw)

    # ------------------------ Click → roiSelected ------------------------ #
    _attach_click_handlers_intensity(widget, plot, active_rois)


def _add_intensity_colorbar_to_widget(
    widget: _SingleWellGraphWidget,
    vmin: float,
    vmax: float,
) -> None:
    """Add a ColorBarItem to the intensity heatmap widget layout."""
    # Create ColorBarItem
    widget.colorbar = pg.ColorBarItem(
        values=(vmin, vmax),
        colorMap=pg.colormap.get("viridis"),
        width=15,
        label="Intensity (dec ΔF/F)",
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)


def _attach_click_handlers_intensity(
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
        idx = round(y)
        if 0 <= idx < len(active_roi_labels):
            widget.roiSelected.emit(str(active_roi_labels[idx]))

    old_click = plot.property("intensity_heatmap_click_handler")
    if old_click is not None:
        try:
            scene.sigMouseClicked.disconnect(old_click)
        except (TypeError, RuntimeError):
            pass

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("intensity_heatmap_click_handler", _on_mouse_clicked)
