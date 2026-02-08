from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import disconnect_hover_handlers
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# PLOT STYLE CONSTANTS
INFERRED_TRACE_COLOR = "k"
INFERRED_TRACE_WIDTH = 3
DFF_OVERLAY_COLOR = "magenta"
DFF_OVERLAY_WIDTH = 3
THRESHOLD_COLOR = "magenta"
THRESHOLD_WIDTH = 3


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
    den_dff: bool = False,
    thresholds: bool = False,
    thresholded: bool = False,
    rising_edges: bool = False,
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
    den_dff : bool
        Optionally overlay denoised ΔF/F traces
    thresholds : bool
        Show spike detection thresholds (only if single ROI selected)
    thresholded : bool
        Plot binarized (0/1) spike traces as vertical lines
    rising_edges : bool
        Mark rising edges of thresholded spikes with vertical lines
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    # Reset ViewBox settings that might have been set by raster plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.invertY(False)  # Ensure Y is not inverted for trace plots
    vb.setAspectLocked(False)  # Ensure aspect ratio is not locked

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

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
        plot.setLabel("left", "Inferred Spikes (a.u.)")
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
        plot.setLabel("left", "Inferred Spikes (a.u.)")
        return

    # ---------------- Global percentiles (for normalization) ---------------- #
    p1 = p2 = 0.0
    if normalize:
        all_values: list[float] = []
        for _roi, traces, data_analysis in roi_data:
            if data_analysis and traces.inferred_spikes:
                spike_data = traces.inferred_spikes
                all_values.extend([float(s) for s in spike_data])

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

        # Get spike data as continuous values
        spike_data = np.asarray(traces.inferred_spikes, dtype=float)

        # x-axis = frames
        x = np.arange(spike_data.size, dtype=float)

        # For thresholded or rising_edges plots, compute binary data
        if thresholded or rising_edges:
            threshold = data_analysis.inferred_spikes_threshold
            if threshold is None or threshold <= 0:
                # Skip this ROI if no valid threshold
                continue

            # Create binary mask where spikes exceed threshold
            binary_spikes = (spike_data > threshold).astype(float)

            if rising_edges:
                # Detect rising edges (transitions from 0 to 1)
                edges = np.diff(binary_spikes, prepend=0) > 0
                curve = _plot_spike_rising_edges(
                    plot=plot,
                    roi_key=str(roi.label_value),
                    x=x,
                    edges=edges,
                    normalize=normalize,
                    index=count,
                    n_rois=n_rois,
                )
            else:
                # Plot thresholded spikes as vertical lines (amplitude-based)
                curve = _plot_thresholded_spikes(
                    plot=plot,
                    roi_key=str(roi.label_value),
                    x=x,
                    spike_data=spike_data,
                    threshold=threshold,
                    normalize=normalize,
                    index=count,
                    n_rois=n_rois,
                    p1=p1,
                    p2=p2,
                )

            if curve is not None:
                curves.append(curve)
        else:
            # Original continuous trace plotting
            # Main spikes curve
            # When den_dff overlay is enabled, force white for spike traces
            curve = _plot_spike_trace(
                plot=plot,
                roi_key=str(roi.label_value),
                x=x,
                trace=spike_data,
                normalize=normalize,
                index=count,
                n_rois=1 if den_dff else n_rois,  # Force white when den_dff=True
                p1=p1,
                p2=p2,
                thresholds=thresholds,
                spikes_threshold=data_analysis.inferred_spikes_threshold,
            )
            if curve is not None:
                curves.append(curve)

            # Optional overlay of denoised ΔF/F
            if den_dff and traces.den_dff:
                dec_trace = np.asarray(traces.den_dff, dtype=float)
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
                        pen=pg.mkPen(DFF_OVERLAY_COLOR, width=DFF_OVERLAY_WIDTH),
                    )

        last_trace = list(traces.inferred_spikes)
        count += 1

    _set_graph_title_and_labels_pg(
        plot, normalize, raw, den_dff, thresholded, rising_edges
    )
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
    pen: pg.QtGui.QPen | None = None,
) -> pg.PlotDataItem | None:
    """Plot inferred spikes trace in pyqtgraph, optionally normalized & stacked."""
    if trace.size == 0:
        return None

    if pen is None:
        # Choose color based on number of ROIs
        pen = pg.mkPen(INFERRED_TRACE_COLOR, width=INFERRED_TRACE_WIDTH)
        # if n_rois == 1:
        #     pen = pg.mkPen("k", width=2)
        # else:
        #     color = pg.intColor(index, hues=max(n_rois, 16))
        #     pen = pg.mkPen(color, width=2)

    if normalize:
        # Reverse offset: lower index (ROI 1) gets higher offset → appears at top
        offset = (n_rois - 1 - index) * 1.1
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
                THRESHOLD_COLOR,
                style=pg.QtCore.Qt.PenStyle.DashLine,
                width=THRESHOLD_WIDTH,
            ),
        )
        line.setZValue(10)
        plot.addItem(line)

    return curve


def _plot_thresholded_spikes(
    plot: pg.PlotItem,
    roi_key: str,
    x: np.ndarray,
    spike_data: np.ndarray,
    threshold: float,
    normalize: bool,
    index: int,
    n_rois: int,
    p1: float,
    p2: float,
) -> pg.PlotDataItem | None:
    """Plot thresholded spikes as vertical lines with amplitude heights.

    Only plots spikes above threshold, with line heights corresponding to
    spike amplitudes.

    Parameters
    ----------
    plot : pg.PlotItem
        The plot item to add the lines to
    roi_key : str
        ROI label/key for identification
    x : np.ndarray
        Frame indices
    spike_data : np.ndarray
        Array of spike amplitudes
    threshold : float
        Spike detection threshold
    normalize : bool
        Whether to stack ROIs vertically
    index : int
        ROI index for stacking
    n_rois : int
        Total number of ROIs
    p1 : float
        Lower percentile for normalization
    p2 : float
        Upper percentile for normalization

    Returns
    -------
    pg.PlotDataItem | None
        Invisible curve for click handling
    """
    if spike_data.size == 0:
        return None

    # Find where spikes exceed threshold
    spike_indices = np.where(spike_data > threshold)[0]

    if len(spike_indices) == 0:
        return None

    # Use black color for all traces
    color = pg.mkPen("k", width=2)

    # Calculate offset for normalization
    if normalize:
        offset = (n_rois - 1 - index) * 1.1
    else:
        offset = 0.0

    # Build all vertical lines as a single PlotDataItem with NaN separators
    # This is much more efficient than creating individual items
    x_lines = []
    y_lines = []

    for spike_idx in spike_indices:
        spike_amp = spike_data[spike_idx]

        if normalize:
            # Normalize the spike amplitude
            denom = p2 - p1
            if denom > 0:
                spike_amp_norm = (spike_amp - p1) / denom
                spike_amp_norm = float(np.clip(spike_amp_norm, 0.0, 1.0))
            else:
                spike_amp_norm = 0.0
            y_bottom = offset
            y_top = offset + spike_amp_norm
        else:
            # Use actual amplitude
            y_bottom = 0.0
            y_top = float(spike_amp)

        # Add this vertical line segment
        x_lines.extend([x[spike_idx], x[spike_idx], np.nan])
        y_lines.extend([y_bottom, y_top, np.nan])

    # Plot all lines as a single clickable item
    if x_lines:
        curve = plot.plot(
            x_lines,
            y_lines,
            pen=color,
            connect="finite",
            name=f"ROI {roi_key}",
        )
        curve.setProperty("roi_label", roi_key)
        curve.setProperty("roi_index", index)
        return curve

    return None


def _plot_spike_rising_edges(
    plot: pg.PlotItem,
    roi_key: str,
    x: np.ndarray,
    edges: np.ndarray,
    normalize: bool,
    index: int,
    n_rois: int,
) -> pg.PlotDataItem | None:
    """Plot rising edges of thresholded spikes as vertical lines.

    Parameters
    ----------
    plot : pg.PlotItem
        The plot item to add the lines to
    roi_key : str
        ROI label/key for identification
    x : np.ndarray
        Frame indices
    edges : np.ndarray
        Boolean array indicating rising edge positions
    normalize : bool
        Whether to stack ROIs vertically
    index : int
        ROI index for stacking
    n_rois : int
        Total number of ROIs

    Returns
    -------
    pg.PlotDataItem | None
        Invisible curve for click handling
    """
    if edges.size == 0:
        return None

    # Find where rising edges occur
    edge_indices = np.where(edges)[0]

    if len(edge_indices) == 0:
        return None

    # Use black color for all traces
    color = pg.mkPen("k", width=2)

    if normalize:
        # Stack ROIs vertically with reverse offset
        offset = (n_rois - 1 - index) * 1.1
        y_bottom = offset
        y_top = offset + 1.0
    else:
        # All edges from 0 to 1
        offset = 0.0
        y_bottom = 0.0
        y_top = 1.0

    # Build all vertical lines as a single PlotDataItem with NaN separators
    # This is much more efficient than creating individual items
    x_lines = []
    y_lines = []

    for edge_idx in edge_indices:
        # Add this vertical line segment
        x_lines.extend([x[edge_idx], x[edge_idx], np.nan])
        y_lines.extend([y_bottom, y_top, np.nan])

    # Plot all lines as a single clickable item
    if x_lines:
        curve = plot.plot(
            x_lines,
            y_lines,
            pen=color,
            connect="finite",
            name=f"ROI {roi_key}",
        )
        curve.setProperty("roi_label", roi_key)
        curve.setProperty("roi_index", index)
        return curve

    return None


def _normalize_trace_percentile(trace: np.ndarray, p1: float, p2: float) -> np.ndarray:
    """Normalize a trace using p1th-p2th percentile, clipped to [0, 1]."""
    tr = np.asarray(trace, dtype=float)
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


def _set_graph_title_and_labels_pg(
    plot: pg.PlotItem,
    normalize: bool,
    raw: bool,
    den_dff: bool,
    thresholded: bool = False,
    rising_edges: bool = False,
) -> None:
    """Set axis labels based on the plotted data."""
    # Initialize defaults
    title = "Inferred Spikes"
    y_lbl = "Spike Amplitude (a.u.)"

    if thresholded:
        if rising_edges:
            if normalize:
                title = "Normalized Inferred Spikes Thresholded (Rising Edges)"
            else:
                title = "Inferred Spikes Thresholded (Rising Edges)"
            y_lbl = "ROI" if normalize else "Rising Edge Events"
        else:
            if normalize:
                title = "Normalized Inferred Spikes Thresholded"
            else:
                title = "Inferred Spikes Thresholded"
            y_lbl = "ROI" if normalize else "Spike Amplitude (a.u.)"
    elif den_dff:
        title = "Denoised ΔF/F & Inferred Spikes"
        y_lbl = "Denoised ΔF/F & Inferred Spikes (a.u.)"
    else:
        title = "Normalized Inferred Spikes" if normalize else "Inferred Spikes"
        y_lbl = "ROI" if normalize else "Inferred Spikes (a.u.)"

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
