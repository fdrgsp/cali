from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


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


def _plot_amplitude_and_frequency_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    amp: bool = False,
    freq: bool = False,
) -> None:
    """Plot amplitude and/or frequency summary data using pyqtgraph."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        cali_logger.warning("No run_id provided for amplitude/frequency plot.")
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "ROIs")
        plot.setLabel("left", "Amplitude / Frequency")
        return

    # Query database for ROI + DataAnalysis
    with Session(engine) as session:
        stmt = (
            select(ROI, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
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
        roi_data: list[tuple[ROI, DataAnalysis]] = session.exec(stmt).all()

    if not roi_data:
        plot.setTitle("No ROI analysis data found for this FOV.")
        plot.setLabel("bottom", "ROIs")
        plot.setLabel("left", "Amplitude / Frequency")
        return

    # ---- Collect data depending on amp/freq flags ----
    x_vals: list[float] = []
    y_vals: list[float] = []
    yerr_vals: list[float] = []
    roi_labels: list[int] = []
    gray_points_x: list[float] = []
    gray_points_y: list[float] = []

    if amp and freq:
        # Amplitude vs frequency (mean ± SEM vs frequency)
        for roi, da in roi_data:
            if not da.peaks_amplitudes_dec_dff or da.dec_dff_frequency is None:
                continue

            amps = np.asarray(da.peaks_amplitudes_dec_dff, dtype=float)
            if amps.size == 0:
                continue

            mean_amp = float(np.mean(amps))
            if amps.size > 1:
                std_amp = float(np.std(amps, ddof=1))
                sem_amp = std_amp / np.sqrt(amps.size)
            else:
                sem_amp = 0.0

            x_vals.append(float(da.dec_dff_frequency))
            y_vals.append(mean_amp)
            yerr_vals.append(sem_amp)
            roi_labels.append(roi.label_value)

        if not x_vals:
            plot.setTitle("No amplitude/frequency data available.")
            plot.setLabel("bottom", "Frequency (Hz)")
            plot.setLabel("left", "Amplitude (dec ΔF/F)")
            return

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        yerr_arr = np.asarray(yerr_vals, dtype=float)

        # Determine colors based on number of ROIs
        n_rois = len(roi_labels)
        if n_rois == 1:
            colors = ["w"]
        else:
            colors = [pg.intColor(i, hues=max(n_rois, 16)) for i in range(n_rois)]

        # Error bars
        err_item = pg.ErrorBarItem(
            x=x_arr,
            y=y_arr,
            top=yerr_arr,
            bottom=yerr_arr,
            beam=0.05 * (x_arr.max() - x_arr.min() if x_arr.size > 1 else 1.0),
            pen=pg.mkPen("w", width=1),
        )
        plot.addItem(err_item)

        # Points (store roi_label in data) - colored per ROI
        scatter = pg.ScatterPlotItem(
            x=x_arr,
            y=y_arr,
            pen=[pg.mkPen(c) for c in colors],
            brush=[pg.mkBrush(c) for c in colors],
            size=7,
            data=[str(lbl) for lbl in roi_labels],
        )
        plot.addItem(scatter)

        _set_graph_title_and_labels_pg(plot, amp=amp, freq=freq)
        _attach_click_handlers_amp_freq(widget, scatter)

    elif amp:
        # Amplitude-only: x = per-ROI index, hide numeric ticks, show unit on Y
        for idx, (roi, da) in enumerate(roi_data):
            if not da.peaks_amplitudes_dec_dff:
                continue

            amps = np.asarray(da.peaks_amplitudes_dec_dff, dtype=float)
            if amps.size == 0:
                continue

            mean_amp = float(np.mean(amps))
            if amps.size > 1:
                std_amp = float(np.std(amps, ddof=1))
                sem_amp = std_amp / np.sqrt(amps.size)
            else:
                sem_amp = 0.0

            x = float(idx)
            x_vals.append(x)
            y_vals.append(mean_amp)
            yerr_vals.append(sem_amp)
            roi_labels.append(roi.label_value)

            # gray background points (individual amplitudes)
            gray_points_x.extend([x] * amps.size)
            gray_points_y.extend(amps.tolist())

        if not x_vals:
            plot.setTitle("No amplitude data available.")
            plot.setLabel("bottom", "ROIs")
            plot.setLabel("left", "Amplitude (dec ΔF/F)")
            return

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        yerr_arr = np.asarray(yerr_vals, dtype=float)

        # Determine colors based on number of ROIs
        n_rois = len(roi_labels)
        if n_rois == 1:
            colors = ["w"]
        else:
            colors = [pg.intColor(i, hues=max(n_rois, 16)) for i in range(n_rois)]

        # Gray individual points
        if gray_points_x:
            gray_scatter = pg.ScatterPlotItem(
                x=np.asarray(gray_points_x, dtype=float),
                y=np.asarray(gray_points_y, dtype=float),
                pen=None,
                brush=pg.mkBrush(150, 150, 150, 160),
                size=5,
            )
            plot.addItem(gray_scatter)

        # Error bars for mean ± SEM
        err_item = pg.ErrorBarItem(
            x=x_arr,
            y=y_arr,
            top=yerr_arr,
            bottom=yerr_arr,
            beam=0.2,
            pen=pg.mkPen("w", width=1),
        )
        plot.addItem(err_item)

        # Mean points with roi_label in data - colored per ROI
        scatter = pg.ScatterPlotItem(
            x=x_arr,
            y=y_arr,
            pen=[pg.mkPen(c) for c in colors],
            brush=[pg.mkBrush(c) for c in colors],
            size=7,
            data=[str(lbl) for lbl in roi_labels],
        )
        plot.addItem(scatter)

        _set_graph_title_and_labels_pg(plot, amp=amp, freq=freq)
        _attach_click_handlers_amp_freq(widget, scatter)

        # Hide numeric x tick labels (keep axis label "ROIs")
        axis = plot.getAxis("bottom")
        axis.setTicks([])
        axis.setStyle(showValues=False)

    elif freq:
        # Frequency-only: x = ROI index, y = dec_dff_frequency
        for idx, (roi, da) in enumerate(roi_data):
            if da.dec_dff_frequency is None:
                continue
            x_vals.append(float(idx))
            y_vals.append(float(da.dec_dff_frequency))
            roi_labels.append(roi.label_value)

        if not x_vals:
            plot.setTitle("No frequency data available.")
            plot.setLabel("bottom", "ROIs")
            plot.setLabel("left", "Frequency (Hz)")
            return

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)

        # Determine colors based on number of ROIs
        n_rois = len(roi_labels)
        if n_rois == 1:
            colors = ["w"]
        else:
            colors = [pg.intColor(i, hues=max(n_rois, 16)) for i in range(n_rois)]

        scatter = pg.ScatterPlotItem(
            x=x_arr,
            y=y_arr,
            pen=[pg.mkPen(c) for c in colors],
            brush=[pg.mkBrush(c) for c in colors],
            size=7,
            data=[str(lbl) for lbl in roi_labels],
        )
        plot.addItem(scatter)

        _set_graph_title_and_labels_pg(plot, amp=amp, freq=freq)
        _attach_click_handlers_amp_freq(widget, scatter)

        # Hide numeric x tick labels (keep axis label "ROIs")
        axis = plot.getAxis("bottom")
        axis.setTicks([])
        axis.setStyle(showValues=False)
    else:
        # Nothing requested
        plot.setTitle("Nothing to plot (amp=False, freq=False).")
        plot.setLabel("bottom", "ROIs")
        plot.setLabel("left", "Amplitude / Frequency")
        return

    # Auto range
    plot.getViewBox().enableAutoRange(x=True, y=True)


def _set_graph_title_and_labels_pg(
    plot: pg.PlotItem,
    amp: bool,
    freq: bool,
) -> None:
    """Set axis labels based on the plotted data (pyqtgraph version)."""
    title = ""
    x_lbl = ""
    y_lbl = ""

    if amp and freq:
        title = (
            "ROIs Mean Calcium Peaks Amplitude ± SEM vs Frequency (Deconvolved ΔF/F)"
        )
        x_lbl = "Frequency (Hz)"
        y_lbl = "Amplitude (dec ΔF/F)"
    elif amp:
        title = "Calcium Peaks Mean Amplitude ± SEM (Deconvolved ΔF/F)"
        x_lbl = "ROIs"
        y_lbl = "Amplitude (dec ΔF/F)"
    elif freq:
        title = "Calcium Peaks Frequency (Deconvolved ΔF/F)"
        x_lbl = "ROIs"
        y_lbl = "Frequency (Hz)"

    plot.setTitle(title)
    plot.setLabel("bottom", x_lbl)
    plot.setLabel("left", y_lbl)


def _attach_click_handlers_amp_freq(
    widget: _SingleWellGraphWidget,
    scatter: pg.ScatterPlotItem,
) -> None:
    """Click on a point → emit widget.roiSelected(str(label))."""

    def _on_clicked(item: pg.ScatterPlotItem, points: list[pg.SpotItem]) -> None:
        if not points:
            return
        data = points[0].data()
        if data is not None:
            widget.roiSelected.emit(str(data))

    scatter.sigClicked.connect(_on_clicked)
