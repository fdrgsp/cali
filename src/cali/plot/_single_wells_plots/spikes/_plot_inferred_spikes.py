from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Helpers: retrieval from ROI histories (kept as in your original file)
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
# Main plotting: inferred spikes (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_inferred_spikes(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    raw: bool = False,
    normalize: bool = False,
    active_only: bool = False,
    dec_dff: bool = False,
    thresholds: bool = False,
) -> None:
    """Plot inferred spikes data by querying database directly (pyqtgraph).

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on (expects .plot_item: pg.PlotItem)
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    run_id : int | None
        The CaliResult.id of the selected run. If provided, only data from this run
        will be plotted.
    raw : bool
        Plot raw inferred spikes (values > 0)
    normalize : bool
        Normalize spike traces globally using percentiles
    active_only : bool
        Only plot active ROIs
    dec_dff : bool
        Optionally overlay deconvolved ΔF/F traces
    thresholds : bool
        Show spike detection thresholds (only if single ROI selected)
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # thresholds only if a single ROI is selected
    thresholds = thresholds if rois and len(rois) == 1 else False

    # Hide shared legend if present (we'll rely on click-to-select instead)
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        cali_logger.warning("No run_id provided for inferred spikes plot.")
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Inferred Spikes")
        return

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

        if active_only:
            stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data: list[tuple[ROI, Traces, DataAnalysis]] = session.exec(stmt).all()

    if not roi_data:
        plot.setTitle("No ROI spike data found for this FOV.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Inferred Spikes")
        return

    # ---------------- Global percentiles (for normalization) ---------------- #
    p1 = p2 = 0.0
    if normalize:
        all_values: list[float] = []
        for _roi, traces, data_analysis in roi_data:
            if data_analysis and traces.inferred_spikes:
                spike_data = traces.inferred_spikes
                if raw:
                    # Raw: all > 0
                    spike_values = [float(s) for s in spike_data if s > 0]
                else:
                    # Thresholded: > threshold
                    the = data_analysis.inferred_spikes_threshold or 0.0
                    spike_values = [float(s) for s in spike_data if s > the]
                all_values.extend(spike_values)

        if all_values:
            p1, p2 = map(float, np.percentile(all_values, [5, 100]))
        else:
            p1, p2 = 0.0, 1.0

    # ------------------------ Plot traces ------------------------ #
    curves: list[pg.PlotDataItem] = []
    count = 0
    rois_rec_time: list[float] = []
    last_trace: list[float] | None = None
    n_rois = len(roi_data)

    for roi, traces, data_analysis in roi_data:
        if data_analysis is None or not traces.inferred_spikes:
            continue

        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        # Raw vs thresholded spikes
        if raw:
            spike_data = np.array(
                [float(s) if s > 0 else 0.0 for s in traces.inferred_spikes],
                dtype=float,
            )
        else:
            the = data_analysis.inferred_spikes_threshold or 0.0
            spike_data = np.array(
                [float(s) if s > the else 0.0 for s in traces.inferred_spikes],
                dtype=float,
            )

        # x-axis = frames
        x = np.arange(spike_data.size, dtype=float)

        # Main spikes curve
        # When dec_dff overlay is enabled, force white for spike traces
        curve = _plot_spike_trace(
            plot=plot,
            roi_key=str(roi.label_value),
            x=x,
            trace=spike_data,
            normalize=normalize,
            index=count,
            n_rois=1 if dec_dff else n_rois,  # Force white when dec_dff=True
            p1=p1,
            p2=p2,
            thresholds=thresholds,
            spikes_threshold=data_analysis.inferred_spikes_threshold,
        )
        if curve is not None:
            curves.append(curve)

        # Optional overlay of deconvolved ΔF/F
        if dec_dff and traces.dec_dff:
            dec_trace = np.asarray(traces.dec_dff, dtype=float)
            if dec_trace.size == spike_data.size:
                _plot_spike_trace(
                    plot=plot,
                    roi_key=str(roi.label_value),
                    x=x,
                    trace=dec_trace,
                    normalize=normalize,
                    index=count,
                    n_rois=n_rois,
                    p1=p1,
                    p2=p2,
                    thresholds=False,
                    spikes_threshold=None,
                    pen=pg.mkPen("y", width=1),
                )

        last_trace = list(traces.inferred_spikes)
        count += 1

    _set_graph_title_and_labels_pg(plot, normalize, raw)
    total_frames = len(last_trace) if last_trace is not None else 1
    _update_time_axis_pg_for_spikes(plot, rois_rec_time, total_frames)

    # Y axis behavior
    y_axis = plot.getAxis("left")
    if normalize:
        # ROIs stacked on Y → hide numeric labels (like elsewhere)
        y_axis.setTicks([])
        y_axis.setStyle(showValues=False)
    else:
        # actual amplitudes on Y → show labels
        y_axis.setStyle(showValues=True)
        y_axis.setTicks(None)

    plot.getViewBox().enableAutoRange(x=True, y=True)

    # Click → roiSelected
    _attach_click_handlers_spikes(widget, curves)


def _plot_spike_trace(
    plot: pg.PlotItem,
    roi_key: str,
    x: np.ndarray,
    trace: np.ndarray,
    normalize: bool,
    index: int,
    n_rois: int,
    p1: float,
    p2: float,
    thresholds: bool = False,
    spikes_threshold: float | None = None,
    pen: pg.mkPen = None,
) -> pg.PlotDataItem | None:
    """Plot inferred spikes trace in pyqtgraph, optionally normalized & stacked."""
    if trace.size == 0:
        return None

    if pen is None:
        # Choose color based on number of ROIs
        if n_rois == 1:
            pen = pg.mkPen("w", width=1)
        else:
            color = pg.intColor(index, hues=max(n_rois, 16))
            pen = pg.mkPen(color, width=1)

    if normalize:
        offset = index * 1.1  # vertical offset per ROI
        tr_norm = _normalize_trace_percentile(trace, p1, p2) + offset
        y = tr_norm
    else:
        y = trace
        offset = 0.0

    curve = plot.plot(
        x,
        y,
        pen=pen,
        name=f"ROI {roi_key}",
    )
    curve.setProperty("roi_label", roi_key)
    curve.setProperty("roi_index", index)

    # Threshold line (only if single ROI selected and thresholds=True)
    if thresholds and spikes_threshold is not None and spikes_threshold > 0.0:
        if normalize:
            denom = p2 - p1
            if denom > 0:
                the_norm = (spikes_threshold - p1) / denom
                the_norm = float(np.clip(the_norm, 0.0, 1.0) + offset)
            else:
                the_norm = offset
            y_the = the_norm
        else:
            y_the = float(spikes_threshold)

        line = pg.InfiniteLine(
            pos=y_the,
            angle=0,
            pen=pg.mkPen(
                "k",
                style=pg.QtCore.Qt.PenStyle.DashLine,
                width=2,
            ),
        )
        line.setZValue(10)
        plot.addItem(line)

    return curve


def _normalize_trace_percentile(trace: np.ndarray, p1: float, p2: float) -> np.ndarray:
    """Normalize a trace using p1th-p2th percentile, clipped to [0, 1]."""
    tr = np.asarray(trace, dtype=float)
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


def _set_graph_title_and_labels_pg(
    plot: pg.PlotItem, normalize: bool, raw: bool
) -> None:
    """Set axis labels based on the plotted data (pyqtgraph version)."""
    title = "Normalized Inferred Spikes" if normalize else "Inferred Spikes"
    title += " (Raw)" if raw else " (Thresholded Spike Data)"
    y_lbl = "ROIs" if normalize else "Inferred Spikes (magnitude)"

    plot.setTitle(title)
    plot.setLabel("left", y_lbl)


def _update_time_axis_pg_for_spikes(
    plot: pg.PlotItem,
    rois_rec_time: list[float],
    total_frames: int,
) -> None:
    """Update the time axis based on recording time (pyqtgraph)."""
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


def _attach_click_handlers_spikes(
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
                widget.roiSelected.emit(str(roi_label))

        curve.sigClicked.connect(_on_curve_clicked)


# -----------------------------------------------------------------------------#
# Normalized spikes + global bursts (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_inferred_spikes_normalized_with_bursts(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot normalized inferred spikes with superimposed *global* burst periods.

    Network bursts are always computed from ALL active ROIs for the given run
    (global network activity). The ROI selection only affects which traces are
    drawn, not how bursts are defined.
    """
    plot = widget.plot_item
    assert plot is not None

    if run_id is None:
        plot.clear()
        cali_logger.warning(
            "No run_id provided for inferred spikes normalized with bursts plot."
        )
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Inferred Spikes")
        return

    # ------------- Burst detection (GLOBAL, ignore ROI subset) -------------#
    from cali.plot._single_wells_plots.burst._plot_inferred_spike_burst_activity import (  # noqa: E501
        _detect_population_bursts,
        _get_burst_parameters,
        _get_population_spike_data,
    )

    bursts: list[tuple[int, int]] = []

    # Use global ROI set for burst parameters and population data
    burst_params = _get_burst_parameters(engine, fov_name, rois=None, run_id=run_id)
    if burst_params is not None:
        burst_threshold, min_burst_duration, smoothing_sigma = burst_params

        spike_trains_array, _, _time_axis = _get_population_spike_data(
            engine, fov_name, rois=None, run_id=run_id
        )

        if spike_trains_array is not None:
            population_activity = np.mean(spike_trains_array, axis=0)

            # Smooth before detection
            if smoothing_sigma > 0:
                smoothed_activity = gaussian_filter1d(
                    population_activity, sigma=smoothing_sigma, mode="nearest"
                )
            else:
                smoothed_activity = population_activity

            # Detect bursts (threshold passed as fraction, not %)
            bursts = _detect_population_bursts(
                smoothed_activity, burst_threshold / 100.0, min_burst_duration
            )

    # -------------------- Plot normalized spikes (subset) -------------------#
    _plot_inferred_spikes(
        widget,
        engine,
        fov_name,
        rois,
        run_id=run_id,
        raw=False,
        normalize=True,
        active_only=False,
        dec_dff=False,
        thresholds=False,
    )

    # ------------------------ Overlay global bursts ------------------------ #
    if bursts:
        plot = widget.plot_item
        assert plot is not None

        for _i, (start, end) in enumerate(bursts):
            region = pg.LinearRegionItem(
                values=(start, end),
                brush=pg.mkBrush(0, 255, 0, 50),  # translucent green
                pen=pg.mkPen(None),  # Remove border lines
                movable=False,
            )
            region.setZValue(-5)  # behind the traces
            plot.addItem(region)
