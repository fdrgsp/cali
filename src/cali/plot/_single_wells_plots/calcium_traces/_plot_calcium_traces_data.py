from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


P1 = 5
P2 = 100
# max number of time points we will draw per trace (automatic downsampling)
MAX_POINTS = 4000


# -----------------------------------------------------------------------------#
# Public entry point (used by registry)
# -----------------------------------------------------------------------------#
def _plot_traces_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
    raw: bool = False,
    dff: bool = False,
    dec: bool = False,
    normalize: bool = False,
    with_peaks: bool = False,
    active_only: bool = False,
    thresholds: bool = False,
) -> None:
    """Plot traces data by querying database directly (pyqtgraph version)."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    # Reset ViewBox settings that might have been set by raster plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)

    # thresholds only if exactly 1 ROI is selected
    thresholds = thresholds if rois and len(rois) == 1 else False

    # --- 1) Get data from DB -----
    data = _get_traces_and_metadata(
        engine,
        fov_name,
        run_id=run_id,
        rois=rois,
        raw=raw,
        dff=dff,
        dec=dec,
        active_only=active_only,
    )
    if data is None:
        return

    Y, labels, data_analysis_list, rois_rec_time = data
    _, T_orig = Y.shape

    # --- 2) Downsample in time if very long ---
    stride = 1
    if T_orig > MAX_POINTS:
        stride = int(np.ceil(T_orig / MAX_POINTS))

    x_full = np.arange(T_orig, dtype=float)
    x = x_full[::stride]
    Y = Y[:, ::stride]  # shape (n_rois, T_ds)
    T_ds = Y.shape[1]

    # --- 3) Normalize + offsets ---
    Y, offsets, p1, p2 = _normalize_and_offset(Y, normalize)

    # --- 4) Draw traces ---
    curves = _draw_traces(plot, x, Y, labels, offsets)

    # --- 5) Peaks + thresholds overlays ---
    if with_peaks:
        _draw_peaks_and_thresholds(
            plot=plot,
            Y=Y,
            offsets=offsets,
            da_list=data_analysis_list,
            labels=labels,
            T_orig=T_orig,
            T_ds=T_ds,
            stride=stride,
            x=x,
            normalize=normalize,
            p1=p1,
            p2=p2,
            thresholds=thresholds,
        )

    # --- 6) Titles, axes & ranges ---
    _set_graph_title_and_labels_pg(plot, dff, dec, normalize, with_peaks)
    # Use first (downsampled) trace as example for time axis
    _update_time_axis_pg(plot, rois_rec_time, traces_example=Y[0], T_orig=T_orig)

    # Y axis behaviour:
    y_axis = plot.getAxis("left")
    if normalize:
        # ROIs stacked on Y → hide numeric labels
        y_axis.setTicks([])
        y_axis.setStyle(showValues=False)
    else:
        # actual amplitudes on Y → show labels
        y_axis.setStyle(showValues=True)
        y_axis.setTicks(None)  # let pyqtgraph auto-generate ticks

    # Make sure everything is visible
    plot.getViewBox().enableAutoRange(x=True, y=True)

    # --- 7) Click handling → roiSelected ---
    _attach_click_handlers(widget, curves)


# -----------------------------------------------------------------------------#
# Data extraction helpers
# -----------------------------------------------------------------------------#
def _get_traces_and_metadata(
    engine: Engine,
    fov_name: str,
    *,
    run_id: int,
    rois: list[int] | None,
    raw: bool,
    dff: bool,
    dec: bool,
    active_only: bool,
) -> tuple[np.ndarray, list[str], list[DataAnalysis | None], list[float]] | None:
    """Query DB and return (Y, labels, da_list, rois_rec_time)."""
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

        if active_only:
            stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data = session.exec(stmt).all()

    if not roi_data:
        return None

    traces: list[np.ndarray] = []
    labels: list[str] = []
    da_list: list[DataAnalysis | None] = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, da in roi_data:
        if not trace_obj:
            continue

        trace = _get_trace(raw, dff, dec, trace_obj)
        if trace is None:
            continue

        tr = np.asarray(trace, dtype=float)
        if tr.size == 0:
            continue

        traces.append(tr)
        labels.append(str(roi_model.label_value))
        da_list.append(da)

        if da is not None and da.total_recording_time_sec is not None:
            rois_rec_time.append(da.total_recording_time_sec)

    if not traces:
        return None

    Y = np.vstack(traces)
    Y = np.nan_to_num(Y, nan=0.0)
    return Y, labels, da_list, rois_rec_time


def _get_trace(
    raw: bool, dff: bool, dec: bool, trace_obj: Traces
) -> list[float] | np.ndarray | None:
    trace = None
    if raw:
        trace = trace_obj.raw_trace
    elif dff:
        trace = trace_obj.dff
        if trace is not None:
            # convert to percent ΔF/F
            trace = np.array(trace) * 100.0
    elif dec:
        trace = trace_obj.dec_dff
    return trace


# -----------------------------------------------------------------------------#
# Normalization + offsets
# -----------------------------------------------------------------------------#
def _normalize_and_offset(
    Y: np.ndarray, normalize: bool
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Apply global percentile normalization and compute vertical offsets."""
    p1 = p2 = 0.0
    if normalize:
        flat = Y.ravel()
        if flat.size > 0:
            p1, p2 = np.percentile(flat, [P1, P2])
            denom = p2 - p1
            if denom == 0:
                Y[:] = 0.0
            else:
                Y = np.clip((Y - p1) / denom, 0, 1)
        else:
            Y[:] = 0.0
            p1, p2 = 0.0, 1.0

        offsets = np.arange(Y.shape[0], dtype=float) * 1.1
    else:
        offsets = np.zeros(Y.shape[0], dtype=float)

    return Y, offsets, float(p1), float(p2)


# -----------------------------------------------------------------------------#
# Drawing traces + overlays
# -----------------------------------------------------------------------------#
def _draw_traces(
    plot: pg.PlotItem,
    x: np.ndarray,
    Y: np.ndarray,
    labels: list[str],
    offsets: np.ndarray,
) -> list[pg.PlotDataItem]:
    """Draw one colored curve per ROI and attach metadata via setProperty."""
    curves: list[pg.PlotDataItem] = []
    n_rois = Y.shape[0]

    for i in range(n_rois):
        y_i = Y[i] + offsets[i]
        roi_label = labels[i]

        # ---- Choose color ----
        if n_rois == 1:
            # Single trace → white
            color = "w"  # or (255, 255, 255) or pg.mkColor("white")
        else:
            # Multi-trace → distinct colors
            color = pg.intColor(i, hues=max(n_rois, 16))

        pen = pg.mkPen(color, width=1)

        curve = plot.plot(
            x,
            y_i,
            pen=pen,
            name=f"ROI {roi_label}",
        )
        # Attach metadata in a non-private way
        curve.setProperty("roi_label", roi_label)
        curve.setProperty("roi_index", i)
        curves.append(curve)

    return curves


def _draw_peaks_and_thresholds(
    plot: pg.PlotItem,
    Y: np.ndarray,
    offsets: np.ndarray,
    da_list: list[DataAnalysis | None],
    labels: list[str],
    T_orig: int,
    T_ds: int,
    stride: int,
    x: np.ndarray,
    normalize: bool,
    p1: float,
    p2: float,
    thresholds: bool,
) -> None:
    """Draw peaks markers and (optionally) thresholds."""
    # Peaks markers
    for i, da in enumerate(da_list):
        if not (da and da.peaks_dec_dff):
            continue

        peaks_indices = np.asarray(da.peaks_dec_dff, dtype=int)
        # clip to original length
        peaks_indices = peaks_indices[(peaks_indices >= 0) & (peaks_indices < T_orig)]
        if peaks_indices.size == 0:
            continue

        # map to downsampled indices
        if stride > 1:
            peaks_ds = (peaks_indices / stride).astype(int)
            peaks_ds = np.clip(peaks_ds, 0, T_ds - 1)
        else:
            peaks_ds = peaks_indices

        y_i = Y[i] + offsets[i]
        plot.plot(
            x[peaks_ds],
            y_i[peaks_ds],
            pen=None,
            symbol="o",
            symbolBrush=pg.mkBrush("yellow"),
            symbolSize=5,
        )

    # Thresholds only if single ROI case
    if not (thresholds and len(labels) == 1):
        return

    da = da_list[0]
    if not da:
        return

    offset0 = float(offsets[0])

    # Peaks height threshold
    if da.peaks_height_dec_dff is not None:
        ph = float(da.peaks_height_dec_dff)
        if normalize:
            denom = (p2 - p1) if (p2 - p1) != 0 else 1.0
            ph_norm = np.clip((ph - p1) / denom, 0, 1) + offset0
            y_the = ph_norm
        else:
            y_the = ph

        line = pg.InfiniteLine(
            pos=y_the,
            angle=0,
            pen=pg.mkPen(
                "orange",
                style=pg.QtCore.Qt.PenStyle.DashLine,
                width=3,
            ),
        )
        line.setZValue(10)  # keep on top
        plot.addItem(line)

    # Prominence threshold bar
    if da.peaks_prominence_dec_dff is not None:
        pp = float(da.peaks_prominence_dec_dff)
        if normalize:
            denom = (p2 - p1) if (p2 - p1) != 0 else 1.0
            y0 = np.clip((0.0 - p1) / denom, 0, 1) + offset0
            y1 = np.clip((pp - p1) / denom, 0, 1) + offset0
        else:
            y0, y1 = 0.0, pp

        plot.plot(
            [x[0], x[0]],
            [y0, y1],
            pen=pg.mkPen("orange", width=3),
        )


# -----------------------------------------------------------------------------#
# Click handling
# -----------------------------------------------------------------------------#
def _attach_click_handlers(
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


# -----------------------------------------------------------------------------#
# Titles / axes helpers
# -----------------------------------------------------------------------------#
def _set_graph_title_and_labels_pg(
    plot: pg.PlotItem,
    dff: bool,
    dec: bool,
    normalize: bool,
    with_peaks: bool,
) -> None:
    if dff:
        title = (
            "Normalized Calcium Traces (ΔF/F)" if normalize else "Calcium Traces (ΔF/F)"
        )
        y_lbl = "ROI" if normalize else "ΔF/F (%)"
    elif dec:
        title = (
            "Normalized Calcium Traces (Deconvolved ΔF/F)"
            if normalize
            else "Calcium Traces (Deconvolved ΔF/F)"
        )
        y_lbl = "ROI" if normalize else "Deconvolved ΔF/F (a.u.)"
    else:
        title = "Normalized Calcium Traces" if normalize else "Raw Calcium Traces"
        y_lbl = "ROI" if normalize else "Fluorescence (a.u.)"
    if with_peaks:
        title += " with Peaks"

    plot.setTitle(title)
    plot.setLabel("left", y_lbl)


def _update_time_axis_pg(
    plot: pg.PlotItem,
    rois_rec_time: list[float],
    traces_example: np.ndarray | list[float] | None,
    T_orig: int,
) -> None:
    """Configure bottom axis as time in seconds if recording time is available."""
    if traces_example is None or not rois_rec_time or sum(rois_rec_time) <= 0:
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
