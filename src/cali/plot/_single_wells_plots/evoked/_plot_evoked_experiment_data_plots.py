from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.plot._util import disconnect_hover_handlers
from cali.sqlmodel._model import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    Traces,
)

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


P1 = 5
P2 = 100
MAX_POINTS = 4000  # downsampling cap like other PG plots

# PLOT STYLE CONSTANTS
DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"
TRACES_WIDTH = 3
AMPLITUDE_ALL_COLOR = (150, 150, 150, 160)  # light gray
AMPLITUDE_ALL_SIZE = 5
PEAKS_SYMBOL = "x"
PEAKS_SYMBOL_SIZE = 10
PEAKS_SYMBOL_COLOR = "k"
LED_COLOR = (0, 0, 255, 200)
LED_SYMBOL = "s"
LED_SYMBOL_SIZE = 8
RASTER_SYMBOL_SIZE = 3
RASTER_SYMBOL_SIZE_LEGEND = 8
RASTER_SYMBOL = "s"
ERROR_BAR_X_WIDTH_MULTIPLIER = 0.02  # fraction of x-range
ERROR_BAR_WIDTH = 2
SCATTER_SIZE = 7


# -----------------------------------------------------------------------------#
# Small helpers (local versions for this module)
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
# Dispatcher
# -----------------------------------------------------------------------------#
def _plot_evoked_experiment_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated_area: bool = False,  # kept for API compatibility; ignored here
    with_rois: bool = False,  # kept for API compatibility; ignored here
    with_peaks: bool = False,
) -> None:
    """
    Main dispatcher for evoked experiment plots in this module.

    Spatial / stimulated-area visualization has been moved to another file.
    Here we only handle the dec dF/F trace plots (with optional peaks).
    """
    _plot_stimulated_vs_non_stimulated_roi_traces(
        widget=widget,
        engine=engine,
        fov_name=fov_name,
        rois=rois,
        run_id=run_id,
        with_peaks=with_peaks,
    )


# -----------------------------------------------------------------------------#
# Peak amplitudes per ROI (stim vs non-stim)  - pyqtgraph version
# -----------------------------------------------------------------------------#


def _plot_stim_and_non_stim_peaks_amplitude(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot both stimulated and non-stimulated peak amplitudes side by side.

    Stimulated ROIs on the left (green), non-stimulated on the right (magenta).
    Each group shows mean ± SEM with individual amplitude points.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.setTitle("Peak Amplitudes\nNo analysis run selected. Please select a run.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    # Query both stimulated and non-stimulated ROIs
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
        results = session.exec(stmt).all()

    if not results:
        plot.setTitle("Peak Amplitudes\nNo ROI data found.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    # Separate data by stimulation status
    stim_data: list[
        tuple[int, float, float, list[float]]
    ] = []  # (label, mean, sem, amps)
    non_stim_data: list[tuple[int, float, float, list[float]]] = []

    for roi_model, _traces, data_analysis in results:
        if not (data_analysis and data_analysis.peaks_amplitudes_dec_dff):
            continue

        amps = np.asarray(data_analysis.peaks_amplitudes_dec_dff, dtype=float)
        if amps.size == 0:
            continue

        mean_amp = float(np.mean(amps))
        if amps.size > 1:
            std_amp = float(np.std(amps, ddof=1))
            sem_amp = std_amp / np.sqrt(amps.size)
        else:
            sem_amp = 0.0

        data_tuple = (roi_model.label_value, mean_amp, sem_amp, amps.tolist())

        if roi_model.stimulated:
            stim_data.append(data_tuple)
        else:
            non_stim_data.append(data_tuple)

    if not stim_data and not non_stim_data:
        plot.setTitle("Peak Amplitudes\nNo peak amplitude data available.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    # Plot stimulated group (left side)
    all_roi_labels: list[int] = []
    x_offset = 0.0

    if stim_data:
        for idx, (roi_label, mean_amp, sem_amp, amps) in enumerate(stim_data):
            all_roi_labels.append(roi_label)
            x = x_offset + idx

            # Individual amplitudes (light gray)
            scatter = pg.ScatterPlotItem(
                x=[x] * len(amps),
                y=amps,
                pen=None,
                brush=pg.mkBrush(AMPLITUDE_ALL_COLOR),
                size=AMPLITUDE_ALL_SIZE,
            )
            plot.addItem(scatter)

            # Error bar
            if sem_amp > 0:
                err = pg.ErrorBarItem(
                    x=np.array([x]),
                    y=np.array([mean_amp]),
                    top=np.array([sem_amp]),
                    bottom=np.array([sem_amp]),
                    beam=0.2,
                    pen=pg.mkPen(STIMULATED_COLOR, width=ERROR_BAR_WIDTH),
                )
                plot.addItem(err)

            # Mean point
            mean_scatter = pg.ScatterPlotItem(
                x=[x],
                y=[mean_amp],
                pen=pg.mkPen(STIMULATED_COLOR, width=1),
                brush=pg.mkBrush(STIMULATED_COLOR),
                size=SCATTER_SIZE,
            )
            plot.addItem(mean_scatter)

        x_offset += len(stim_data) + 1  # Gap between groups

    # Plot non-stimulated group (right side)
    if non_stim_data:
        for idx, (roi_label, mean_amp, sem_amp, amps) in enumerate(non_stim_data):
            all_roi_labels.append(roi_label)
            x = x_offset + idx

            # Individual amplitudes (light gray)
            scatter = pg.ScatterPlotItem(
                x=[x] * len(amps),
                y=amps,
                pen=None,
                brush=pg.mkBrush(AMPLITUDE_ALL_COLOR),
                size=AMPLITUDE_ALL_SIZE,
            )
            plot.addItem(scatter)

            # Error bar
            if sem_amp > 0:
                err = pg.ErrorBarItem(
                    x=np.array([x]),
                    y=np.array([mean_amp]),
                    top=np.array([sem_amp]),
                    bottom=np.array([sem_amp]),
                    beam=0.2,
                    pen=pg.mkPen(NON_STIMULATED_COLOR, width=ERROR_BAR_WIDTH),
                )
                plot.addItem(err)

            # Mean point
            mean_scatter = pg.ScatterPlotItem(
                x=[x],
                y=[mean_amp],
                pen=pg.mkPen(NON_STIMULATED_COLOR, width=1),
                brush=pg.mkBrush(NON_STIMULATED_COLOR),
                size=SCATTER_SIZE,
            )
            plot.addItem(mean_scatter)

    # Axis labels and styling
    plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
    plot.setLabel("bottom", "Stimulated → Non-Stimulated ROIs")

    # Y-axis: show tick values
    y_axis = plot.getAxis("left")
    y_axis.setStyle(showValues=True)

    # X-axis: no numeric tick labels
    x_axis = plot.getAxis("bottom")
    x_axis.setTicks([])
    x_axis.setStyle(showValues=False)

    # Enable autorange for proper scaling
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.enableAutoRange(x=True, y=True)

    # Title with counts
    plot.setTitle(
        f"Peak Amplitudes (Stimulated: {len(stim_data)} ROIs, "
        f"Non-Stimulated: {len(non_stim_data)} ROIs)"
    )

    # Legend
    legend = getattr(widget, "legend", None)
    if legend is not None:
        legend.clear()
        if stim_data:
            stim_item = pg.ScatterPlotItem(
                pen=pg.mkPen(STIMULATED_COLOR, width=1),
                brush=pg.mkBrush(STIMULATED_COLOR),
                size=SCATTER_SIZE,
            )
            legend.addItem(stim_item, "Stimulated ROIs")
        if non_stim_data:
            non_stim_item = pg.ScatterPlotItem(
                pen=pg.mkPen(NON_STIMULATED_COLOR, width=1),
                brush=pg.mkBrush(NON_STIMULATED_COLOR),
                size=SCATTER_SIZE,
            )
            legend.addItem(non_stim_item, "Non-Stimulated ROIs")
        legend.setVisible(True)

    # Store ROI labels for click mapping
    plot.setProperty("peaks_amp_roi_labels", all_roi_labels)
    _attach_click_handlers_peaks_amp(widget, plot)


def _attach_click_handlers_peaks_amp(
    widget: _SingleWellGraphWidget, plot: pg.PlotItem
) -> None:
    """Map mouse click x to nearest ROI in peaks amplitude plot."""
    from pyqtgraph import Point

    scene = plot.scene()
    vb = plot.getViewBox()

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        p: Point = vb.mapSceneToView(pos)
        x = float(p.x())
        roi_labels: list[int] | None = plot.property("peaks_amp_roi_labels")
        if not roi_labels:
            return
        idx = round(x)
        if 0 <= idx < len(roi_labels):
            widget.roiSelected.emit(str(roi_labels[idx]))

    # disconnect previous handler if any
    old_click = plot.property("peaks_amp_click_handler")
    if old_click is not None:
        try:
            scene.sigMouseClicked.disconnect(old_click)
        except (TypeError, RuntimeError):
            pass

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("peaks_amp_click_handler", _on_mouse_clicked)


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#
def extract_leading_number(key: str) -> float:
    """Extract leading number from key (before '_'), stripping units if present."""
    if match := re.match(r"(\d+(?:\.\d+)?)", key.split("_")[0]):
        return float(match[1])
    raise ValueError(f"Could not extract a valid number from key: {key}")


# -----------------------------------------------------------------------------#
# dec ΔF/F traces: stimulated vs non-stimulated (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_stimulated_vs_non_stimulated_roi_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_peaks: bool = False,
) -> None:
    """Plot dec ΔF/F traces with global percentile normalization (5th-100th).

    - Stimulated ROIs: green
    - Non-stimulated ROIs: magenta
    - Traces are stacked vertically.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous content
    plot.clear()

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    vb.enableAutoRange(x=True, y=True)

    if run_id is None:
        plot.setTitle("Stimulated vs Non-Stimulated ROIs (Normalized)\nNo run selected")
        plot.setLabel("bottom", "Time (s)")
        plot.setLabel("left", "ROIs (stacked)")
        return

    # ---------- DB QUERY ----------
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
        results = session.exec(stmt).all()

    if not results:
        plot.setTitle("Stimulated vs Non-Stimulated ROIs (Normalized)\nNo ROI data")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROIs (stacked)")
        return

    # ---------- SPLIT BY STIM STATUS ----------
    stimulated_data: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    non_stimulated_data: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.dec_dff:
            if roi_model.stimulated:
                stimulated_data.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_data.append((roi_model, trace_obj, data_analysis))
            if data_analysis and data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    # ---------- GLOBAL PERCENTILE NORMALIZATION ----------
    all_values: list[float] = []
    for _, trace_obj, _ in results:
        if trace_obj and trace_obj.dec_dff:
            all_values.extend(trace_obj.dec_dff)

    if all_values:
        percentiles = np.percentile(all_values, [P1, P2])
        p1, p2 = float(percentiles[0]), float(percentiles[1])
    else:
        p1, p2 = 0.0, 1.0

    # Collect traces as arrays
    stim_traces: list[np.ndarray] = []
    stim_labels: list[int] = []
    stim_peaks: list[list[int]] = []

    for roi_model, trace_obj, data_analysis in stimulated_data:
        if not trace_obj.dec_dff:
            continue
        tr_norm = np.asarray(
            _normalize_trace_percentile(trace_obj.dec_dff, p1, p2), dtype=float
        )
        if tr_norm.size == 0:
            continue
        stim_traces.append(tr_norm)
        stim_labels.append(roi_model.label_value)
        if data_analysis and data_analysis.peaks_dec_dff:
            stim_peaks.append([int(p) for p in data_analysis.peaks_dec_dff])
        else:
            stim_peaks.append([])

    non_traces: list[np.ndarray] = []
    non_labels: list[int] = []
    non_peaks: list[list[int]] = []

    for roi_model, trace_obj, data_analysis in non_stimulated_data:
        if not trace_obj.dec_dff:
            continue
        tr_norm = np.asarray(
            _normalize_trace_percentile(trace_obj.dec_dff, p1, p2), dtype=float
        )
        if tr_norm.size == 0:
            continue
        non_traces.append(tr_norm)
        non_labels.append(roi_model.label_value)
        if data_analysis and data_analysis.peaks_dec_dff:
            non_peaks.append([int(p) for p in data_analysis.peaks_dec_dff])
        else:
            non_peaks.append([])

    if not stim_traces and not non_traces:
        plot.setTitle(
            "Stimulated vs Non-Stimulated ROIs (Normalized)\nNo valid dec ΔF/F traces"
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROIs (stacked)")
        return

    # ---------- STACK & DOWNSAMPLE ----------
    example_trace = (stim_traces or non_traces)[0]
    T_orig = example_trace.size

    stride = 1
    if T_orig > MAX_POINTS:
        stride = int(np.ceil(T_orig / MAX_POINTS))

    x_full = np.arange(T_orig, dtype=float)
    x = x_full[::stride]

    curves: list[pg.PlotDataItem] = []
    last_raw_trace: list[float] | None = None

    # Stimulated group
    if stim_traces:
        Y_stim = np.vstack(stim_traces)
        Y_stim = Y_stim[:, ::stride]
        n_stim, T_ds = Y_stim.shape
        # Reverse offsets: ROI 1 (index 0) gets highest offset → appears at top
        offsets_stim = np.arange(n_stim - 1, -1, -1, dtype=float) * 1.1

        for i in range(n_stim):
            y_i = Y_stim[i] + offsets_stim[i]
            roi_label = stim_labels[i]
            curve = plot.plot(
                x,
                y_i,
                pen=pg.mkPen(STIMULATED_COLOR, width=TRACES_WIDTH),
                name=f"ROI {roi_label}",
            )
            curve.setProperty("roi_label", str(roi_label))
            curve.setProperty("roi_index", i)
            curves.append(curve)

            if with_peaks and stim_peaks[i]:
                peaks = np.asarray(stim_peaks[i], dtype=int)
                peaks = peaks[(peaks >= 0) & (peaks < T_orig)]
                if peaks.size > 0:
                    if stride > 1:
                        peaks_ds = (peaks / stride).astype(int)
                        peaks_ds = np.clip(peaks_ds, 0, T_ds - 1)
                    else:
                        peaks_ds = peaks
                    plot.plot(
                        x[peaks_ds],
                        y_i[peaks_ds],
                        pen=None,
                        symbol=PEAKS_SYMBOL,
                        symbolBrush=pg.mkBrush(PEAKS_SYMBOL_COLOR),
                        symbolSize=PEAKS_SYMBOL_SIZE,
                    )

        first_stim_trace = stimulated_data[0][1].dec_dff
        if first_stim_trace:
            last_raw_trace = list(first_stim_trace)

    # Non-stimulated group
    base_offset = len(stim_traces)
    if non_traces:
        Y_non = np.vstack(non_traces)
        Y_non = Y_non[:, ::stride]
        n_non, T_ds2 = Y_non.shape
        # Reverse offsets: continue from base_offset, but reversed within group
        offsets_non = (np.arange(n_non - 1, -1, -1, dtype=float) + base_offset) * 1.1

        for i in range(n_non):
            y_i = Y_non[i] + offsets_non[i]
            roi_label = non_labels[i]
            curve = plot.plot(
                x,
                y_i,
                pen=pg.mkPen(NON_STIMULATED_COLOR, width=TRACES_WIDTH),
                name=f"ROI {roi_label}",
            )
            curve.setProperty("roi_label", str(roi_label))
            curve.setProperty("roi_index", base_offset + i)
            curves.append(curve)

            if with_peaks and non_peaks[i]:
                peaks = np.asarray(non_peaks[i], dtype=int)
                peaks = peaks[(peaks >= 0) & (peaks < T_orig)]
                if peaks.size > 0:
                    if stride > 1:
                        peaks_ds = (peaks / stride).astype(int)
                        peaks_ds = np.clip(peaks_ds, 0, T_ds2 - 1)
                    else:
                        peaks_ds = peaks
                    plot.plot(
                        x[peaks_ds],
                        y_i[peaks_ds],
                        pen=None,
                        symbol=PEAKS_SYMBOL,
                        symbolBrush=pg.mkBrush(PEAKS_SYMBOL_COLOR),
                        symbolSize=PEAKS_SYMBOL_SIZE,
                    )

        if last_raw_trace is None:
            first_non_trace = non_stimulated_data[0][1].dec_dff
            if first_non_trace:
                last_raw_trace = list(first_non_trace)

    # ---------- AXES & TITLES ----------
    plot.setTitle("Stimulated vs Non-Stimulated ROIs (Normalized dec ΔF/F)")
    plot.setLabel("left", "ROIs (stacked)")
    _update_time_axis_pg(plot, rois_rec_time, last_raw_trace, T_orig=T_orig)

    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    # ---------- LEGEND ----------
    legend = getattr(widget, "legend", None)
    if legend is not None:
        legend.clear()

        # Add legend items for stimulated and non-stimulated traces
        if stim_traces:
            stim_item = pg.PlotDataItem(
                pen=pg.mkPen(STIMULATED_COLOR, width=TRACES_WIDTH)
            )
            legend.addItem(stim_item, "Stimulated ROIs")

        if non_traces:
            non_stim_item = pg.PlotDataItem(
                pen=pg.mkPen(NON_STIMULATED_COLOR, width=TRACES_WIDTH)
            )
            legend.addItem(non_stim_item, "Non-Stimulated ROIs")

        if with_peaks:
            peak_item = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(PEAKS_SYMBOL_COLOR),
                size=PEAKS_SYMBOL_SIZE,
                symbol=PEAKS_SYMBOL,
            )
            legend.addItem(peak_item, "Peaks")

        # Add LED stimulation legend item
        led_item = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(LED_COLOR),
            size=LED_SYMBOL_SIZE,
            symbol=LED_SYMBOL,
        )
        legend.addItem(led_item, "LED Stimulation")

        legend.setVisible(True)

    vb.enableAutoRange(x=True, y=True)

    # ---------- LED STIMULATION BANDS ----------
    # Get frame rate from data analysis
    frame_rate = None
    for _, _, data_analysis in results:
        if data_analysis and data_analysis.total_recording_time_sec is not None:
            frame_rate = T_orig / data_analysis.total_recording_time_sec
            break

    _add_led_stimulation_bands(plot, engine, run_id, frame_rate, stride)

    # ---------- CLICK → roiSelected ----------
    _attach_click_handlers_evoked(widget, curves)


# -----------------------------------------------------------------------------#
# Spike raster: stimulated vs non-stimulated (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_stimulated_vs_non_stimulated_spike_raster(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot raster of thresholded spikes (green=stim, magenta=non-stim) with pg.

    Each suprathreshold burst in the inferred spike trace is collapsed to a
    single event (rising edge), so the raster shows one tick per inferred spike
    event rather than one tick per frame above threshold.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    vb.invertY(True)

    # Hide legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    from cali.sqlmodel._model import FOV as FOVModel  # avoid name clash

    if run_id is None:
        plot.setTitle(
            "Stimulated vs Non-Stimulated Spike Raster Plot\nNo run selected."
        )
        plot.setLabel("bottom", "Frames")
        return

    # ------------------------ Query DB ------------------------ #
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOVModel, ROI.fov_id == FOVModel.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOVModel.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        plot.setTitle("Spike Raster\nNo active ROI data found for this FOV.")
        plot.setLabel("bottom", "Frames")
        return

    stimulated_rois: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    non_stimulated_rois: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    active_roi_labels: list[int] = []
    rois_rec_time: list[float] = []
    total_frames = 0

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.inferred_spikes is not None and data_analysis:
            active_roi_labels.append(roi_model.label_value)
            if roi_model.stimulated:
                stimulated_rois.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_rois.append((roi_model, trace_obj, data_analysis))

            if data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

            total_frames = max(total_frames, len(trace_obj.inferred_spikes))

    if not stimulated_rois and not non_stimulated_rois:
        plot.setTitle("Spike Raster\nNo spike data available.")
        plot.setLabel("bottom", "Frames")
        return

    # ------------------------ Build raster ------------------------ #
    y_row = 0

    def _add_raster_row(
        roi_model: ROI,
        trace_obj: Traces,
        data_analysis: DataAnalysis | None,
        color: str,
        row_index: int,
    ) -> bool:
        """Threshold + rising-edge detection; return True if anything was plotted."""
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if spikes.size == 0:
            return False

        the = (
            float(data_analysis.inferred_spikes_threshold)
            if data_analysis and data_analysis.inferred_spikes_threshold is not None
            else 0.0
        )

        above = spikes > the
        if not np.any(above):
            return False

        # rising edges: 0 -> 1 transitions (collapse runs of 1s to a single event)
        rising = above & ~np.concatenate(([False], above[:-1]))
        spike_indices = np.where(rising)[0]
        if spike_indices.size == 0:
            return False

        item = pg.ScatterPlotItem(
            x=spike_indices.astype(float),
            y=np.full_like(spike_indices, row_index, dtype=float),
            pen=None,
            brush=pg.mkBrush(color),
            size=RASTER_SYMBOL_SIZE,
            symbol=RASTER_SYMBOL,
        )
        item.setProperty("roi_label", str(roi_model.label_value))
        plot.addItem(item)
        return True

    # Stimulated rows
    for roi_model, trace_obj, data_analysis in stimulated_rois:
        _add_raster_row(roi_model, trace_obj, data_analysis, STIMULATED_COLOR, y_row)
        y_row += 1

    # Non-stimulated rows
    for roi_model, trace_obj, data_analysis in non_stimulated_rois:
        _add_raster_row(
            roi_model, trace_obj, data_analysis, NON_STIMULATED_COLOR, y_row
        )
        y_row += 1

    plot.setTitle("Stimulated vs Non-Stimulated Spike Raster Plot (Thresholded)")
    plot.setLabel("left", "ROI")
    _update_time_axis_pg_frames(plot, rois_rec_time, total_frames)

    # hide y tick values (but keep axis label)
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    # ---------- LEGEND ----------
    legend = getattr(widget, "legend", None)
    if legend is not None:
        legend.clear()

        # Add legend items for stimulated and non-stimulated spikes
        if stimulated_rois:
            stim_item = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(STIMULATED_COLOR),
                size=RASTER_SYMBOL_SIZE_LEGEND,
                symbol=RASTER_SYMBOL,
            )
            legend.addItem(stim_item, "Stimulated ROIs")

        if non_stimulated_rois:
            non_stim_item = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(NON_STIMULATED_COLOR),
                size=RASTER_SYMBOL_SIZE_LEGEND,
                symbol=RASTER_SYMBOL,
            )
            legend.addItem(non_stim_item, "Non-Stimulated ROIs")

        # Add LED stimulation legend item
        led_item = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(LED_COLOR),
            size=LED_SYMBOL_SIZE,
            symbol=LED_SYMBOL,
        )
        legend.addItem(led_item, "LED Stimulation")

        legend.setVisible(True)

    # ---------- LED STIMULATION BANDS ----------
    # Get frame rate from data analysis
    frame_rate = None
    for _, _, data_analysis in results:
        if data_analysis and data_analysis.total_recording_time_sec is not None:
            frame_rate = total_frames / data_analysis.total_recording_time_sec
            break

    _add_led_stimulation_bands(plot, engine, run_id, frame_rate, stride=1)

    # Set x-range to full frames with some padding at the end, enable autorange for y
    if total_frames > 0:
        vb.setXRange(0, total_frames * 1.05, padding=0)
    vb.enableAutoRange(x=False, y=True)

    _attach_click_handlers_raster(widget, plot, active_roi_labels)


def _attach_click_handlers_raster(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    active_roi_labels: list[int],
) -> None:
    """Map click y-row to ROI label in raster."""
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

    old_click = plot.property("raster_click_handler")
    if old_click is not None:
        try:
            scene.sigMouseClicked.disconnect(old_click)
        except (TypeError, RuntimeError):
            pass

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("raster_click_handler", _on_mouse_clicked)


# -----------------------------------------------------------------------------#
# Calcium peaks raster: stimulated vs non-stimulated (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_stimulated_vs_non_stimulated_calcium_peaks_raster(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot raster of calcium peaks (green=stim, magenta=non-stim) with pg.

    Each detected calcium peak (from peaks_dec_dff) is shown as a tick mark.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    vb.invertY(True)

    # Hide legend
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    from cali.sqlmodel._model import FOV as FOVModel  # avoid name clash

    if run_id is None:
        plot.setTitle(
            "Stimulated vs Non-Stimulated Calcium Peaks Raster Plot\nNo run selected."
        )
        plot.setLabel("bottom", "Frames")
        return

    # ------------------------ Query DB ------------------------ #
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOVModel, ROI.fov_id == FOVModel.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOVModel.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        plot.setTitle("Calcium Peaks Raster\nNo active ROI data found for this FOV.")
        plot.setLabel("bottom", "Frames")
        return

    stimulated_rois: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    non_stimulated_rois: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    active_roi_labels: list[int] = []
    rois_rec_time: list[float] = []
    total_frames = 0

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.dec_dff is not None and data_analysis:
            active_roi_labels.append(roi_model.label_value)
            if roi_model.stimulated:
                stimulated_rois.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_rois.append((roi_model, trace_obj, data_analysis))

            if data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

            total_frames = max(total_frames, len(trace_obj.dec_dff))

    if not stimulated_rois and not non_stimulated_rois:
        plot.setTitle("Calcium Peaks Raster\nNo calcium trace data available.")
        plot.setLabel("bottom", "Frames")
        return

    # ------------------------ Build raster ------------------------ #
    y_row = 0

    def _add_calcium_raster_row(
        roi_model: ROI,
        trace_obj: Traces,
        data_analysis: DataAnalysis | None,
        color: str,
        row_index: int,
    ) -> bool:
        """Plot calcium peaks; return True if anything was plotted."""
        if not data_analysis or not data_analysis.peaks_dec_dff:
            return False

        peak_indices = np.asarray(data_analysis.peaks_dec_dff, dtype=float)
        if peak_indices.size == 0:
            return False

        item = pg.ScatterPlotItem(
            x=peak_indices,
            y=np.full_like(peak_indices, row_index, dtype=float),
            pen=None,
            brush=pg.mkBrush(color),
            size=RASTER_SYMBOL_SIZE,
            symbol=RASTER_SYMBOL,
        )
        item.setProperty("roi_label", str(roi_model.label_value))
        plot.addItem(item)
        return True

    # Stimulated rows
    for roi_model, trace_obj, data_analysis in stimulated_rois:
        _add_calcium_raster_row(
            roi_model, trace_obj, data_analysis, STIMULATED_COLOR, y_row
        )
        y_row += 1

    # Non-stimulated rows
    for roi_model, trace_obj, data_analysis in non_stimulated_rois:
        _add_calcium_raster_row(
            roi_model, trace_obj, data_analysis, NON_STIMULATED_COLOR, y_row
        )
        y_row += 1

    plot.setTitle("Stimulated vs Non-Stimulated Calcium Peaks Raster Plot")
    plot.setLabel("left", "ROI")
    _update_time_axis_pg_frames(plot, rois_rec_time, total_frames)

    # hide y tick values (but keep axis label)
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    # ---------- LEGEND ----------
    legend = getattr(widget, "legend", None)
    if legend is not None:
        legend.clear()

        # Add legend items for stimulated and non-stimulated peaks
        if stimulated_rois:
            stim_item = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(STIMULATED_COLOR),
                size=RASTER_SYMBOL_SIZE_LEGEND,
                symbol=RASTER_SYMBOL,
            )
            legend.addItem(stim_item, "Stimulated ROIs")

        if non_stimulated_rois:
            non_stim_item = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(NON_STIMULATED_COLOR),
                size=RASTER_SYMBOL_SIZE_LEGEND,
                symbol=RASTER_SYMBOL,
            )
            legend.addItem(non_stim_item, "Non-Stimulated ROIs")

        # Add LED stimulation legend item
        led_item = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(LED_COLOR),
            size=LED_SYMBOL_SIZE,
            symbol=LED_SYMBOL,
        )
        legend.addItem(led_item, "LED Stimulation")

        legend.setVisible(True)

    # ---------- LED STIMULATION BANDS ----------
    # Get frame rate from data analysis
    frame_rate = None
    for _, _, data_analysis in results:
        if data_analysis and data_analysis.total_recording_time_sec is not None:
            frame_rate = total_frames / data_analysis.total_recording_time_sec
            break

    _add_led_stimulation_bands(plot, engine, run_id, frame_rate, stride=1)

    # Set x-range to full frames with some padding at the end, enable autorange for y
    if total_frames > 0:
        vb.setXRange(0, total_frames * 1.05, padding=0)
    vb.enableAutoRange(x=False, y=True)

    _attach_click_handlers_raster(widget, plot, active_roi_labels)


# -----------------------------------------------------------------------------#
# Spike traces: stimulated vs non-stimulated (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_stimulated_vs_non_stimulated_spike_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot continuous inferred spike traces separated by stimulation status."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    vb = plot.getViewBox()
    vb.setAspectLocked(False)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    from cali.sqlmodel._model import FOV as FOVModel  # avoid name clash

    if run_id is None:
        plot.setTitle(
            "Stimulated vs Non-Stimulated Spike Traces\nNo analysis run selected."
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Inferred Spikes (Thresholded)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOVModel, ROI.fov_id == FOVModel.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOVModel.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        plot.setTitle("Stimulated vs Non-Stimulated Spike Traces\nNo ROI data.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Inferred Spikes (Thresholded)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    stimulated_data: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    non_stimulated_data: list[tuple[ROI, Traces, DataAnalysis | None]] = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.inferred_spikes:
            if roi_model.stimulated:
                stimulated_data.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_data.append((roi_model, trace_obj, data_analysis))

            if data_analysis and data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not stimulated_data and not non_stimulated_data:
        plot.setTitle("Stimulated vs Non-Stimulated Spike Traces\nNo spike data.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Inferred Spikes (Thresholded)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    curves: list[pg.PlotDataItem] = []
    count = 0
    total_frames = 0

    # Stim traces
    for roi_model, trace_obj, data_analysis in stimulated_data:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis and data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        offset = count * 1.1
        y = spikes + offset
        x = np.arange(y.size, dtype=float)
        curve = plot.plot(
            x,
            y,
            pen=pg.mkPen(STIMULATED_COLOR, width=TRACES_WIDTH),
            name=f"ROI {roi_model.label_value}",
        )
        curve.setProperty("roi_label", str(roi_model.label_value))
        curve.setProperty("roi_index", count)
        curves.append(curve)
        count += 1
        total_frames = max(total_frames, y.size)

    # Non-stim traces
    for roi_model, trace_obj, data_analysis in non_stimulated_data:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis and data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        offset = count * 1.1
        y = spikes + offset
        x = np.arange(y.size, dtype=float)
        curve = plot.plot(
            x,
            y,
            pen=pg.mkPen(NON_STIMULATED_COLOR, width=TRACES_WIDTH),
            name=f"ROI {roi_model.label_value}",
        )
        curve.setProperty("roi_label", str(roi_model.label_value))
        curve.setProperty("roi_index", count)
        curves.append(curve)
        count += 1
        total_frames = max(total_frames, y.size)

    plot.setLabel("left", "Inferred Spikes (Thresholded)")
    plot.setTitle(
        "Stimulated vs Non-Stimulated Spike Traces\n(Thresholded Inferred Spikes)"
    )

    # hide y tick values, but keep the axis label text
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    # ---------- LEGEND ----------
    legend = getattr(widget, "legend", None)
    if legend is not None:
        legend.clear()

        # Add legend items for stimulated and non-stimulated traces
        if stimulated_data:
            stim_item = pg.PlotDataItem(
                pen=pg.mkPen(STIMULATED_COLOR, width=TRACES_WIDTH)
            )
            legend.addItem(stim_item, "Stimulated ROIs")

        if non_stimulated_data:
            non_stim_item = pg.PlotDataItem(
                pen=pg.mkPen(NON_STIMULATED_COLOR, width=TRACES_WIDTH)
            )
            legend.addItem(non_stim_item, "Non-Stimulated ROIs")

        # Add LED stimulation legend item
        led_item = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(LED_COLOR),
            size=LED_SYMBOL_SIZE,
            symbol=LED_SYMBOL,
        )
        legend.addItem(led_item, "LED Stimulation")

        legend.setVisible(True)

    _update_time_axis_pg_frames(plot, rois_rec_time, total_frames)

    # ---------- LED STIMULATION BANDS ----------
    # Get frame rate from data analysis
    frame_rate = None
    for _, _, data_analysis in results:
        if data_analysis and data_analysis.total_recording_time_sec is not None:
            frame_rate = total_frames / data_analysis.total_recording_time_sec
            break

    _add_led_stimulation_bands(plot, engine, run_id, frame_rate, stride=1)

    _attach_click_handlers_evoked(widget, curves)


# -----------------------------------------------------------------------------#
# Shared helpers
# -----------------------------------------------------------------------------#
def _add_led_stimulation_bands(
    plot: pg.PlotItem,
    engine: Engine,
    run_id: int,
    frame_rate: float | None = None,
    stride: int = 1,
    color: tuple[int, int, int, int] = LED_COLOR,
) -> None:
    """Add vertical bands for LED stimulation events.

    Parameters
    ----------
    plot : pg.PlotItem
        The plot to add bands to
    engine : Engine
        Database engine
    run_id : int
        Analysis result ID to get stimulation settings from
    frame_rate : float | None
        Frame rate in Hz (frames per second). If None, tries to get from settings.
    stride : int
        Downsampling stride used in the plot (default 1)
    color : tuple[int, int, int, int]
        RGBA color tuple for the bands (default blue: 0, 0, 255, 200)
    """
    with Session(engine) as session:
        # Get analysis settings from run_id
        result = session.get(CaliResult, run_id)
        if not result or not result.analysis_settings_id:
            return

        settings = session.get(AnalysisSettings, result.analysis_settings_id)
        if not settings:
            return

        # Check if we have LED pulse information
        if not settings.led_pulse_on_frames or not settings.led_pulse_duration:
            return

        # Use frame rate from settings if not provided
        if frame_rate is None:
            frame_rate = settings.frame_rate

        # Convert LED pulse duration from milliseconds to frames
        pulse_duration_frames = (settings.led_pulse_duration / 1000.0) * frame_rate

        # Add vertical bands for each LED pulse
        for pulse_frame in settings.led_pulse_on_frames:
            # Account for downsampling stride
            start_frame = (pulse_frame - 1) / stride
            end_frame = ((pulse_frame - 1) + pulse_duration_frames) / stride

            # Create a vertical LinearRegionItem for the LED pulse
            region = pg.LinearRegionItem(
                values=(start_frame, end_frame),
                orientation="vertical",
                brush=pg.mkBrush(*color),
                pen=pg.mkPen(None),  # No border
                movable=False,
            )
            plot.addItem(region)


def _normalize_trace_percentile(
    trace: list[float], p1: float, p2: float
) -> list[float]:
    """Normalize a trace using the global p1-p2 percentiles."""
    tr = np.array(trace, dtype=float)
    denom = p2 - p1
    if denom == 0:
        return cast("list[float]", np.zeros_like(tr).tolist())
    normalized = (tr - p1) / denom
    normalized = np.clip(normalized, 0, 1)
    return cast("list[float]", normalized.tolist())


def _update_time_axis_pg(
    plot: pg.PlotItem,
    rois_rec_time: list[float],
    trace: list[float] | None,
    T_orig: int,
) -> None:
    """Configure bottom axis as time in seconds if recording time is available."""
    if trace is None or not rois_rec_time or sum(rois_rec_time) <= 0:
        plot.setLabel("bottom", "Frames")
        return

    total_frames = T_orig
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


def _update_time_axis_pg_frames(
    plot: pg.PlotItem,
    rois_rec_time: list[float],
    total_frames: int,
) -> None:
    """Time axis helper when we know only total_frames (e.g. raster)."""
    if total_frames <= 1 or not rois_rec_time or sum(rois_rec_time) <= 0:
        plot.setLabel("bottom", "Frames")
        return

    avg_rec_time = int(np.mean(rois_rec_time))
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    tick_interval = avg_rec_time / total_frames
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]

    axis = plot.getAxis("bottom")
    axis.setTicks([list(zip(x_ticks.tolist(), x_labels))])
    plot.setLabel("bottom", "Time (s)")


def _attach_click_handlers_evoked(
    widget: _SingleWellGraphWidget, curves: list[pg.PlotDataItem]
) -> None:
    """Make curves clickable and emit widget.roiSelected on click."""
    for curve in curves:
        curve.setCurveClickable(True, 8)

        def _on_curve_clicked(
            curve_obj: pg.PlotCurveItem,
            ev: MouseClickEvent,
            c: pg.PlotDataItem = curve,
        ) -> None:
            roi_label = c.property("roi_label")
            if roi_label is not None:
                widget.roiSelected.emit(roi_label)

        curve.sigClicked.connect(_on_curve_clicked)
