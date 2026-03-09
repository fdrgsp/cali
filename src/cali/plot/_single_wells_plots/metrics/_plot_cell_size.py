from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import disconnect_hover_handlers
from cali.sqlmodel._model import FOV, ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# PLOT STYLE CONSTANTS
SCATTER_SIZE = 7


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


def _plot_cell_size_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot cell size per ROI using pyqtgraph."""
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
        cali_logger.warning("No run_id provided for cell size plot.")
        plot.setTitle(
            "Cell Size per ROI\nNo analysis run selected. Please select a run."
        )
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Cell Size (a.u.)")
        return

    # ---------------------- DB QUERY & PREP DATA ---------------------- #
    roi_labels: list[int] = []
    cell_sizes: list[float] = []
    units = ""

    with Session(engine) as session:
        detection_settings_id: int | None = None

        result = session.get(CaliResult, run_id)
        if result:
            detection_settings_id = result.detection_settings_id

        stmt = select(ROI).join(FOV).where(col(FOV.name) == fov_name)

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        if detection_settings_id is not None:
            stmt = stmt.where(col(ROI.detection_settings_id) == detection_settings_id)

        stmt = stmt.order_by(col(ROI.label_value))
        roi_models = session.exec(stmt).all()

        if not roi_models:
            plot.setTitle("Cell Size per ROI\nNo cell size data found for this FOV.")
            plot.setLabel("bottom", "ROI")
            plot.setLabel("left", "Cell Size (a.u.)")
            return

        # Extract data
        for roi in roi_models:
            if roi.cell_size is None:
                continue

            roi_labels.append(roi.label_value)
            cell_sizes.append(float(roi.cell_size))
            if not units and roi.cell_size_units:
                units = roi.cell_size_units

    if not roi_labels:
        plot.setTitle("Cell Size per ROI\nNo cell size data found for this FOV.")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "Cell Size (a.u.)")
        return

    if not units:
        units = "a.u."

    x_positions = np.arange(len(roi_labels), dtype=float)
    y_values = np.asarray(cell_sizes, dtype=float)

    # Multi-color ScatterPlotItem with one color per ROI
    n_rois = len(roi_labels)
    brushes = [pg.mkBrush(pg.intColor(i, hues=max(n_rois, 16))) for i in range(n_rois)]

    scatter = pg.ScatterPlotItem(
        x=x_positions,
        y=y_values,
        pen=None,
        brush=brushes,
        size=SCATTER_SIZE,
    )
    plot.addItem(scatter)

    # Store ROI labels on plot for click-mapping
    plot.setProperty("cell_size_roi_labels", roi_labels)

    # ---------------------- AXES & TITLE ---------------------- #
    plot.setTitle("Cell Size per ROI")
    plot.setLabel("left", f"Cell Size ({units})")
    plot.setLabel("bottom", "ROI")

    # No numeric tick labels on X (only axis label "ROI")
    x_axis = plot.getAxis("bottom")
    x_axis.setTicks([])  # remove tick labels
    x_axis.setStyle(showValues=False)

    # Let Y axis auto-generate ticks and show labels
    y_axis = plot.getAxis("left")
    y_axis.setStyle(showValues=True)
    y_axis.setTicks(None)

    plot.getViewBox().enableAutoRange(x=True, y=True)

    # ---------------------- CLICK → roiSelected ---------------------- #
    _attach_click_handlers_cell_size(widget, plot)


def _attach_click_handlers_cell_size(
    widget: _SingleWellGraphWidget, plot: pg.PlotItem
) -> None:
    """Map mouse click x position to nearest ROI label in cell-size plot."""
    from pyqtgraph import Point

    scene = plot.scene()
    vb = plot.getViewBox()

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return

        p: Point = vb.mapSceneToView(pos)
        x = float(p.x())

        roi_labels: list[int] | None = plot.property("cell_size_roi_labels")
        if not roi_labels:
            return

        idx = round(x)
        if 0 <= idx < len(roi_labels):
            widget.roiSelected.emit(str(roi_labels[idx]))

    # Disconnect previous handler if present
    old_click = plot.property("cell_size_click_handler")
    if old_click is not None:
        try:
            scene.sigMouseClicked.disconnect(old_click)
        except (TypeError, RuntimeError):
            pass

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("cell_size_click_handler", _on_mouse_clicked)
