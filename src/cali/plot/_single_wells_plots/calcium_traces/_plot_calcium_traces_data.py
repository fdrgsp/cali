from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, col, select

from cali.plot._hover_utils import setup_pick_click
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


COUNT_INCREMENT = 1
P1 = 5
P2 = 100


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
    """Plot traces data by querying database directly (vectorized Matplotlib).

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    engine : Engine
        SQLAlchemy Engine connected to the database
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    run_id : int
        The CaliResult.id of the selected run.
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    raw : bool
        Plot raw traces
    dff : bool
        Plot ΔF/F traces
    dec : bool
        Plot deconvolved ΔF/F traces
    normalize : bool
        Normalize traces using percentile method
    with_peaks : bool
        Show detected peaks
    active_only : bool
        Only plot active ROIs
    thresholds : bool
        Show peak detection thresholds (only if single ROI selected)
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    # show peaks thresholds only if only 1 roi is selected
    thresholds = thresholds if rois and len(rois) == 1 else False

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

        if active_only:
            stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

        stmt = stmt.order_by(col(ROI.label_value))

        roi_data = session.exec(stmt).all()

    if not roi_data:
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ---------- COLLECT TRACES & METADATA ----------
    traces: list[np.ndarray] = []
    labels: list[str] = []
    data_analysis_list: list[DataAnalysis | None] = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in roi_data:
        if not trace_obj:
            continue

        trace = _get_trace(raw, dff, dec, trace_obj)
        if trace is None:
            continue

        tr = np.asarray(trace, dtype=float)
        traces.append(tr)
        labels.append(str(roi_model.label_value))
        data_analysis_list.append(data_analysis)

        if (
            data_analysis is not None
            and data_analysis.total_recording_time_sec is not None
        ):
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not traces:
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # At this point we assume all traces have the same length.
    # If that's not guaranteed, you'd need to pad/crop them.
    Y = np.vstack(traces)  # shape = (n_rois, T)

    # ---------- GLOBAL PERCENTILE NORMALIZATION ----------
    p1 = p2 = 0.0
    if normalize:
        flat = Y.ravel()
        if flat.size > 0:
            p1, p2 = np.percentile(flat, [P1, P2])
            denom = p2 - p1
            if denom == 0:
                Y = np.zeros_like(Y)
            else:
                Y = np.clip((Y - p1) / denom, 0, 1)
        else:
            Y = np.zeros_like(Y)
            p1, p2 = 0.0, 1.0

    # ---------- VERTICAL OFFSETS ----------
    if normalize:
        offsets = np.arange(Y.shape[0])[:, None] * 1.1  # (n_rois, 1)
        Y_plot = Y + offsets
    else:
        Y_plot = Y
        offsets = np.zeros((Y.shape[0], 1))  # for threshold math when not normalized

    # ---------- SINGLE VECTORIZED PLOT CALL ----------
    # ax.plot expects (T, n) for many lines in one call
    lines = ax.plot(Y_plot.T, linewidth=0.8, picker=3)

    # set labels / metadata so hover code can identify ROIs
    for line, roi_label in zip(lines, labels):
        line.set_label(f"ROI {roi_label}")
        # Optional: custom attribute if your hover code wants it
        line._roi_label = roi_label

    # ---------- PEAKS & THRESHOLDS ----------
    if with_peaks:
        # plot peaks as markers
        for i, da in enumerate(data_analysis_list):
            if not (da and da.peaks_dec_dff):
                continue

            peaks_indices = np.asarray(da.peaks_dec_dff, dtype=int)
            ax.plot(
                peaks_indices,
                Y_plot[i, peaks_indices],
                "x",
            )

        # thresholds only if single ROI selected
        if thresholds and len(labels) == 1:
            da = data_analysis_list[0]
            if da:
                roi_label = labels[0]
                offset0 = float(offsets[0, 0])

                if da.peaks_height_dec_dff is not None:
                    ph = float(da.peaks_height_dec_dff)
                    if normalize:
                        denom = (p2 - p1) if (p2 - p1) != 0 else 1.0
                        ph_norm = np.clip((ph - p1) / denom, 0, 1) + offset0
                        y_the = ph_norm
                    else:
                        y_the = ph

                    ax.axhline(
                        y=y_the,
                        color="black",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.6,
                        label=(f"Peaks Height threshold\n(ROI {roi_label} - {ph:.4f})"),
                    )

                if da.peaks_prominence_dec_dff is not None:
                    pp = float(da.peaks_prominence_dec_dff)
                    if normalize:
                        denom = (p2 - p1) if (p2 - p1) != 0 else 1.0
                        y0 = np.clip((0.0 - p1) / denom, 0, 1) + offset0
                        y1 = np.clip((pp - p1) / denom, 0, 1) + offset0
                    else:
                        y0, y1 = 0.0, pp

                    ax.plot(
                        [-3, -3],
                        [y0, y1],
                        color="orange",
                        linestyle="-",
                        linewidth=5,
                        alpha=0.8,
                        label=(
                            f"Peaks Prominence Threshold\n(ROI {roi_label} - {pp:.4f})"
                        ),
                    )

    # ---------- TITLES / AXES / HOVER ----------
    _set_graph_title_and_labels(ax, dff, dec, normalize, with_peaks)

    # any trace length works; all rows have same T
    last_trace = Y[0]
    _update_time_axis(ax, rois_rec_time, last_trace)

    # ---------- HOVER FUNCTIONALITY ----------s
    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_trace(
    raw: bool, dff: bool, dec: bool, trace_obj: Traces
) -> list[float] | np.ndarray | None:
    if dff:
        trace = trace_obj.dff
    elif dec:
        trace = trace_obj.dec_dff
    elif raw:
        trace = trace_obj.raw_trace
    else:
        trace = trace_obj.corrected_trace
    return trace


def _set_graph_title_and_labels(
    ax: Axes,
    dff: bool,
    dec: bool,
    normalize: bool,
    with_peaks: bool,
) -> None:
    """Set axis labels based on the plotted data."""
    if dff:
        title = (
            "Normalized Calcium Traces (ΔF/F)" if normalize else "Calcium Traces (ΔF/F)"
        )
        y_lbl = "ROIs" if normalize else "ΔF/F"
    elif dec:
        title = (
            "Normalized Calcium Traces (Deconvolved ΔF/F)"
            if normalize
            else "Calcium Traces (Deconvolved ΔF/F)"
        )
        y_lbl = "ROIs" if normalize else "Deconvolved ΔF/F"
    else:
        title = "Normalized Calcium Traces" if normalize else "Raw Calcium Traces"
        y_lbl = "ROIs" if normalize else "Fluorescence Intensity"
    if with_peaks:
        title += " with Peaks"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | np.ndarray | None
) -> None:
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    # get the average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    # get total number of frames from the trace
    tr = np.asarray(trace)
    total_frames = tr.shape[-1] if tr is not None else 1
    # compute tick positions
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click(ax, widget, picker_tolerance=3)
