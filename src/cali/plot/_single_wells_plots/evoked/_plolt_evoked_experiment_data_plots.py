from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"
P1 = 5
P2 = 100
MAX_POINTS = 4000  # downsampling cap like other PG plots


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
def _plot_stim_or_not_stim_peaks_amplitude(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated: bool = False,
) -> None:
    """
    Visualize stimulated / non-stimulated peak amplitudes per ROI (mean ± SEM).

    Uses:
        - ErrorBarItem for mean ± SEM
        - Scatter markers for individual peak amplitudes
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # Hide shared legend if present here
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.setTitle("Peak Amplitudes\nNo analysis run selected. Please select a run.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
        # ensure y-axis values are visible for this plot
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

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
            .where(col(ROI.stimulated) == stimulated)
        )

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        kind = "stimulated" if stimulated else "non-stimulated"
        plot.setTitle(f"Peak Amplitudes\nNo {kind} ROI data found.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    roi_labels: list[int] = []
    x_positions: list[float] = []
    means: list[float] = []
    sems: list[float] = []
    all_points_x: list[float] = []
    all_points_y: list[float] = []

    color = STIMULATED_COLOR if stimulated else NON_STIMULATED_COLOR

    for idx, (roi_model, _traces, data_analysis) in enumerate(results):
        if not (data_analysis and data_analysis.peaks_amplitudes_dec_dff):
            continue

        amps = np.asarray(data_analysis.peaks_amplitudes_dec_dff, dtype=float)
        if amps.size == 0:
            continue

        roi_labels.append(roi_model.label_value)
        x = float(idx)
        x_positions.append(x)

        mean_amp = float(np.mean(amps))
        if amps.size > 1:
            std_amp = float(np.std(amps, ddof=1))
            sem_amp = std_amp / np.sqrt(amps.size)
        else:
            sem_amp = 0.0

        means.append(mean_amp)
        sems.append(sem_amp)

        # individual amplitudes
        all_points_x.extend([x] * amps.size)
        all_points_y.extend(amps.tolist())

    if not roi_labels:
        plot.setTitle("Peak Amplitudes\nNo peak amplitude data available.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
        y_axis = plot.getAxis("left")
        y_axis.setStyle(showValues=True)
        return

    x_arr = np.asarray(x_positions, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    sem_arr = np.asarray(sems, dtype=float)

    # Scatter for individual amplitudes (light gray)
    if all_points_x:
        scatter = pg.ScatterPlotItem(
            x=np.asarray(all_points_x, dtype=float),
            y=np.asarray(all_points_y, dtype=float),
            pen=None,
            brush=pg.mkBrush(150, 150, 150, 160),
            size=5,
        )
        plot.addItem(scatter)

    # Error bars for mean ± SEM
    if sem_arr.size > 0:
        err = pg.ErrorBarItem(
            x=x_arr,
            y=means_arr,
            top=sem_arr,
            bottom=sem_arr,
            beam=0.2,
            pen=pg.mkPen(color, width=2),
        )
        plot.addItem(err)

    # Scatter for means (on top, solid color)
    mean_scatter = pg.ScatterPlotItem(
        x=x_arr,
        y=means_arr,
        pen=pg.mkPen(color, width=1),
        brush=pg.mkBrush(color),
        size=7,
    )
    plot.addItem(mean_scatter)

    # Axis labels
    plot.setLabel("left", "Peak Amplitude (dec ΔF/F)")
    plot.setLabel("bottom", "ROI")

    # Y-axis: always show tick values (re-enable if a previous plot hid them)
    y_axis = plot.getAxis("left")
    y_axis.setStyle(showValues=True)

    # X-axis: no numeric tick labels, only the axis label "ROI"
    x_axis = plot.getAxis("bottom")
    x_axis.setTicks([])  # clear ticks
    x_axis.setStyle(showValues=False)

    title = "Stimulated" if stimulated else "Non-Stimulated"
    plot.setTitle(f"{title} ROI Mean Peak Amplitudes ± SEM")

    # store ROI labels for click mapping
    plot.setProperty("peaks_amp_roi_labels", roi_labels)

    # click → nearest ROI on x-axis
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
        offsets_stim = np.arange(n_stim, dtype=float) * 1.1

        for i in range(n_stim):
            y_i = Y_stim[i] + offsets_stim[i]
            roi_label = stim_labels[i]
            curve = plot.plot(
                x,
                y_i,
                pen=pg.mkPen(STIMULATED_COLOR, width=1),
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
                        symbol="o",
                        symbolBrush=pg.mkBrush("yellow"),
                        symbolSize=5,
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
        offsets_non = (np.arange(n_non, dtype=float) + base_offset) * 1.1

        for i in range(n_non):
            y_i = Y_non[i] + offsets_non[i]
            roi_label = non_labels[i]
            curve = plot.plot(
                x,
                y_i,
                pen=pg.mkPen(NON_STIMULATED_COLOR, width=1),
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
                        symbol="o",
                        symbolBrush=pg.mkBrush("yellow"),
                        symbolSize=5,
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

    vb.enableAutoRange(x=True, y=True)

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
    """Plot raster of thresholded spikes (green=stim, magenta=non-stim) with pg."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setAspectLocked(False)

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
        if trace_obj and trace_obj.inferred_spikes and data_analysis:
            active_roi_labels.append(roi_model.label_value)
            if roi_model.stimulated:
                stimulated_rois.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_rois.append((roi_model, trace_obj, data_analysis))

            if data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

            total_frames = max(total_frames, len(trace_obj.inferred_spikes or []))

    if not stimulated_rois and not non_stimulated_rois:
        plot.setTitle("Spike Raster\nNo spike data available.")
        plot.setLabel("bottom", "Frames")
        return

    # Build raster with ScatterPlotItem
    y_row = 0

    # Stim
    for roi_model, trace_obj, data_analysis in stimulated_rois:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis and data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        spike_indices = np.where(spikes > 0.0)[0]
        if spike_indices.size > 0:
            item = pg.ScatterPlotItem(
                x=spike_indices.astype(float),
                y=np.full_like(spike_indices, y_row, dtype=float),
                pen=None,
                brush=pg.mkBrush(STIMULATED_COLOR),
                size=3,
            )
            item.setProperty("roi_label", str(roi_model.label_value))
            plot.addItem(item)
        y_row += 1

    # Non-stim
    for roi_model, trace_obj, data_analysis in non_stimulated_rois:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis and data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        spike_indices = np.where(spikes > 0.0)[0]
        if spike_indices.size > 0:
            item = pg.ScatterPlotItem(
                x=spike_indices.astype(float),
                y=np.full_like(spike_indices, y_row, dtype=float),
                pen=None,
                brush=pg.mkBrush(NON_STIMULATED_COLOR),
                size=3,
            )
            item.setProperty("roi_label", str(roi_model.label_value))
            plot.addItem(item)
        y_row += 1

    plot.setTitle("Stimulated vs Non-Stimulated Spike Raster Plot")
    plot.setLabel("left", "ROI (rows)")
    _update_time_axis_pg_frames(plot, rois_rec_time, total_frames)

    # hide y tick values (but keep axis label)
    y_axis = plot.getAxis("left")
    y_axis.setTicks([])
    y_axis.setStyle(showValues=False)

    vb.enableAutoRange(x=True, y=True)

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
        idx = round(y)
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
            pen=pg.mkPen(STIMULATED_COLOR, width=1.5),
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
            pen=pg.mkPen(NON_STIMULATED_COLOR, width=1.5),
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

    _update_time_axis_pg_frames(plot, rois_rec_time, total_frames)

    _attach_click_handlers_evoked(widget, curves)


# -----------------------------------------------------------------------------#
# Shared helpers
# -----------------------------------------------------------------------------#
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
