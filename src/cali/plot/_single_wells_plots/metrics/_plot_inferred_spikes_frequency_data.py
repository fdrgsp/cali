from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import disconnect_hover_handlers
from cali.sqlmodel._model import FOV, ROI, DataAnalysis

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# PLOT STYLE CONSTANTS
SCATTER_SIZE = 7


def _plot_inferred_spikes_frequency_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    rising_edge: bool = False,
) -> None:
    """Plot inferred spikes frequency data using pyqtgraph.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget to plot on.
    engine : Engine
        SQLAlchemy engine for database access.
    fov_name : str
        Name of the FOV to plot.
    rois : list[int] | None
        Optional list of ROI label values to include.
    run_id : int | None
        Analysis run ID to use.
    rising_edge : bool
        If True, plot rising edge frequency; otherwise plot thresholded frequency.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()

    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        cali_logger.warning("No run_id provided for inferred spikes frequency plot.")
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Frequency (Hz)")
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
        plot.setLabel("left", "Frequency (Hz)")
        return

    # Collect frequency data
    x_vals: list[float] = []
    y_vals: list[float] = []
    roi_labels: list[int] = []

    for idx, (roi, da) in enumerate(roi_data):
        # Get the appropriate frequency based on rising_edge flag
        if rising_edge:
            freq = da.inferred_spikes_rising_edge_frequency
        else:
            freq = da.inferred_spikes_frequency

        if freq is None:
            continue

        x_vals.append(float(idx))
        y_vals.append(float(freq))
        roi_labels.append(roi.label_value)

    if not x_vals:
        freq_type = "rising edge" if rising_edge else "thresholded"
        plot.setTitle(f"No inferred spikes {freq_type} frequency data available.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Frequency (Hz)")
        return

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    # Determine colors based on number of ROIs
    n_rois = len(roi_labels)
    if n_rois == 1:
        colors = ["k"]
    else:
        colors = [pg.intColor(i, hues=max(n_rois, 16)) for i in range(n_rois)]

    scatter = pg.ScatterPlotItem(
        x=x_arr,
        y=y_arr,
        pen=[pg.mkPen(c) for c in colors],
        brush=[pg.mkBrush(c) for c in colors],
        size=SCATTER_SIZE,
        data=[str(lbl) for lbl in roi_labels],
    )
    plot.addItem(scatter)

    # Set title and labels
    if rising_edge:
        title = "Inferred Spikes Rising Edge Frequency"
    else:
        title = "Inferred Spikes Thresholded Frequency"

    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "Frequency (Hz)")

    # Attach click handler
    _attach_click_handlers(widget, scatter)

    # Hide numeric x tick labels (keep axis label "ROI")
    axis = plot.getAxis("bottom")
    axis.setTicks([])
    axis.setStyle(showValues=False)

    # Auto range
    plot.getViewBox().enableAutoRange(x=True, y=True)


def _attach_click_handlers(
    widget: _SingleWellGraphWidget,
    scatter: pg.ScatterPlotItem,
) -> None:
    """Click on a point -> emit widget.roiSelected(str(label))."""

    def _on_clicked(item: pg.ScatterPlotItem, points: list[pg.SpotItem]) -> None:
        if not points:
            return
        data = points[0].data()
        if data is not None:
            widget.roiSelected.emit(str(data))

    scatter.sigClicked.connect(_on_clicked)
