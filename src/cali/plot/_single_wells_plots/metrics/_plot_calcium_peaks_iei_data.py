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
        return roi_model.traces_history[0] if roi_model.traces_history else None
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
        return (
            roi_model.data_analysis_history[0]
            if roi_model.data_analysis_history
            else None
        )
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    return (
        roi_model.data_analysis_history[0] if roi_model.data_analysis_history else None
    )


def _plot_iei_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inter-event interval data by querying the database (pyqtgraph).

    For each ROI:
    - Mean IEI ± SEM as a white point + error bar
    - Individual IEI values as light-gray background points
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        cali_logger.warning("No run_id provided for IEI plot.")
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Inter-Event Interval (s)")
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
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Inter-Event Interval (s)")
        return

    # --- Collect IEI stats per ROI ---
    x_vals: list[float] = []
    y_means: list[float] = []
    y_sem: list[float] = []
    roi_labels: list[int] = []
    gray_x: list[float] = []
    gray_y: list[float] = []

    for idx, (roi, da) in enumerate(roi_data):
        if not da.iei:
            continue

        iei = np.asarray(da.iei, dtype=float)
        if iei.size == 0:
            continue

        # Mean IEI
        mean_iei = float(np.mean(iei))

        # SEM = std / sqrt(N)
        if iei.size > 1:
            std_iei = float(np.std(iei, ddof=1))
            sem_iei = std_iei / np.sqrt(iei.size)
        else:
            sem_iei = 0.0

        x = float(idx)  # internal x-position; we'll hide numeric ticks
        x_vals.append(x)
        y_means.append(mean_iei)
        y_sem.append(sem_iei)
        roi_labels.append(roi.label_value)

        # gray background points (individual IEIs)
        gray_x.extend([x] * iei.size)
        gray_y.extend(iei.tolist())

    if not x_vals:
        plot.setTitle("No inter-event interval data available.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Inter-Event Interval (s)")
        return

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_means, dtype=float)
    sem_arr = np.asarray(y_sem, dtype=float)

    # Determine colors based on number of ROIs
    n_rois = len(roi_labels)
    if n_rois == 1:
        colors = ["k"]
    else:
        colors = [pg.intColor(i, hues=max(n_rois, 16)) for i in range(n_rois)]

    # --- Individual IEIs (gray background) ---
    if gray_x:
        gray_scatter = pg.ScatterPlotItem(
            x=np.asarray(gray_x, dtype=float),
            y=np.asarray(gray_y, dtype=float),
            pen=None,
            brush=pg.mkBrush(150, 150, 150, 160),
            size=5,
        )
        plot.addItem(gray_scatter)

    # --- Error bars for mean ± SEM ---
    err_item = pg.ErrorBarItem(
        x=x_arr,
        y=y_arr,
        top=sem_arr,
        bottom=sem_arr,
        beam=0.2,
        pen=pg.mkPen("k", width=2),
    )
    plot.addItem(err_item)

    # --- Mean points (clickable, with ROI label in data) - colored per ROI ---
    mean_scatter = pg.ScatterPlotItem(
        x=x_arr,
        y=y_arr,
        pen=[pg.mkPen(c) for c in colors],
        brush=[pg.mkBrush(c) for c in colors],
        size=7,
        data=[str(lbl) for lbl in roi_labels],
    )
    plot.addItem(mean_scatter)

    _set_graph_title_and_labels_pg(plot)
    _attach_click_handlers_iei(widget, mean_scatter)

    # Hide numeric x tick labels (keep axis label "ROI")
    axis = plot.getAxis("bottom")
    axis.setTicks([])
    axis.setStyle(showValues=False)

    plot.getViewBox().enableAutoRange(x=True, y=True)


def _set_graph_title_and_labels_pg(plot: pg.PlotItem) -> None:
    """Set axis labels based on the plotted data (pyqtgraph version)."""
    title = "Calcium Peaks Inter-Event Intervals (Mean ± SEM - Deconvolved ΔF/F)"
    plot.setTitle(title)
    plot.setLabel("left", "Inter-Event Interval (s)")
    plot.setLabel("bottom", "ROI")


def _attach_click_handlers_iei(
    widget: _SingleWellGraphWidget,
    scatter: pg.ScatterPlotItem,
) -> None:
    """Click on a mean IEI point → emit widget.roiSelected(str(label))."""

    def _on_clicked(item: pg.ScatterPlotItem, points: list[pg.SpotItem]) -> None:
        if not points:
            return
        data = points[0].data()
        if data is not None:
            widget.roiSelected.emit(str(data))

    scatter.sigClicked.connect(_on_clicked)
