from __future__ import annotations

from typing import TYPE_CHECKING

import cmap  # use the cmap library for grayscale
import numpy as np
import pyqtgraph as pg
from skimage.measure import find_contours
from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI, AnalysisSettings, CaliResult, Mask
from cali.util._util import coordinates_to_mask

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"


def _visualize_stimulated_area(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_rois: bool = False,
    stimulated_area: bool = False,
) -> None:
    """
    Visualize stimulated area with ROI mask overlay using pyqtgraph.

    - if with_rois: show ROI masks, colored by stimulated/non-stim
    - if stimulated_area: overlay the stimulation mask outline
    - else: show simple ROI statistics text
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous content
    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Make sure the view is "image-like"
    vb.invertY(True)  # origin at top-left like images
    vb.setAspectLocked(True)  # keep pixels square

    # Reset shared legend (we'll reuse it for this plot)
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.setTitle(
            "Stimulated Area / ROI Masks\n"
            "No analysis run selected. Please select a run."
        )
        return

    stim_mask = None
    roi_data: list[tuple[ROI, Mask | None]] = []
    image_shape: tuple[int, int] | None = None

    # ------------------ DB QUERY ------------------ #
    with Session(engine) as session:
        session.expire_all()

        # 1) Stimulation mask from AnalysisSettings (via CaliResult)
        result = session.get(CaliResult, run_id)
        if result and result.analysis_settings_id:
            analysis_settings = session.get(
                AnalysisSettings, result.analysis_settings_id
            )
            if analysis_settings and analysis_settings.stimulation_mask_id:
                mask_obj = session.get(Mask, analysis_settings.stimulation_mask_id)
                if (
                    mask_obj
                    and mask_obj.coords_y is not None
                    and mask_obj.coords_x is not None
                    and mask_obj.height is not None
                    and mask_obj.width is not None
                ):
                    coords = (mask_obj.coords_y, mask_obj.coords_x)
                    shape = (mask_obj.height, mask_obj.width)
                    stim_mask = coordinates_to_mask(coords, shape)
                    image_shape = shape

        # 2) ROI + masks
        stmt = (
            select(ROI, Mask)
            .join(FOV, ROI.fov_id == FOV.id)
            .outerjoin(Mask, ROI.roi_mask_id == Mask.id)
            .where(col(FOV.name) == fov_name)
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()
        roi_data = [(roi, mask) for roi, mask in results]

        # If we didn't get shape from stim mask, try any ROI mask
        if image_shape is None and roi_data:
            for _roi, mask in roi_data:
                if mask and mask.height and mask.width:
                    image_shape = (mask.height, mask.width)
                    break

    if not roi_data:
        _display_roi_statistics_pg(plot, [], [])
        return

    # Split counts for stats
    stimulated_roi_labels: list[int] = []
    non_stimulated_roi_labels: list[int] = []
    for roi, _ in roi_data:
        if roi.stimulated:
            stimulated_roi_labels.append(roi.label_value)
        else:
            non_stimulated_roi_labels.append(roi.label_value)

    # We might need the shared legend
    legend = getattr(widget, "legend", None)

    # ------------------ WITH ROI MASKS ------------------ #
    if with_rois and image_shape:
        H, W = image_shape
        labels = np.zeros((H, W), dtype=np.int32)

        # Fill label image (each ROI id as integer)
        for roi, mask in roi_data:
            if (
                mask
                and mask.coords_y is not None
                and mask.coords_x is not None
                and mask.height is not None
                and mask.width is not None
            ):
                roi_mask = coordinates_to_mask(
                    (mask.coords_y, mask.coords_x), (mask.height, mask.width)
                )
                labels[roi_mask] = roi.label_value

        # Convert labels to RGBA image: green = stim, magenta = non-stim
        img_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        img_rgba[..., 3] = 255  # alpha

        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            if lbl == 0:
                continue
            mask_lbl = labels == lbl
            roi = next(r for r, _ in roi_data if r.label_value == lbl)
            if roi.stimulated:
                color = (0, 255, 0, 255)  # green
            else:
                color = (255, 0, 255, 255)  # magenta
            img_rgba[mask_lbl] = color

        img_item = pg.ImageItem(img_rgba, axisOrder="row-major")
        plot.addItem(img_item)

        # Place pixels so centers are at integer coords (0..W-1, 0..H-1)
        rect = pg.QtCore.QRectF(-0.5, -0.5, W, H)
        img_item.setRect(rect)
        vb.setRange(rect, padding=0.0)

        # Overlay stimulation mask contours (yellow)
        if stimulated_area and stim_mask is not None:
            conts = find_contours(stim_mask.astype(float), level=0.5)
            for c in conts:
                plot.plot(
                    c[:, 1],
                    c[:, 0],
                    pen=pg.mkPen("yellow", width=2),
                )

        plot.setTitle("Stimulated / Non-Stimulated ROI Masks")

        # Use shared legend instead of text labels
        if legend is not None:
            legend.clear()

            # Dummy items just to show colors in legend
            stim_item = pg.PlotDataItem(pen=pg.mkPen(STIMULATED_COLOR, width=2))
            non_stim_item = pg.PlotDataItem(pen=pg.mkPen(NON_STIMULATED_COLOR, width=2))
            legend.addItem(stim_item, "Stimulated ROIs")
            legend.addItem(non_stim_item, "Non-Stimulated ROIs")

            if stimulated_area and stim_mask is not None:
                stim_area_item = pg.PlotDataItem(pen=pg.mkPen("yellow", width=2))
                legend.addItem(stim_area_item, "Stimulation Area")

            legend.setVisible(True)

    # ------------------ ONLY STIM AREA ------------------ #
    elif stimulated_area and stim_mask is not None:
        H, W = stim_mask.shape

        img_item = pg.ImageItem(stim_mask.astype(float), axisOrder="row-major")

        # Use cmap.Colormap("gray") → LUT for pyqtgraph
        gray_cmap = cmap.Colormap("gray")
        lut = gray_cmap.lut(N=256, gamma=1.0)  # (256, 4) floats in [0,1]
        lut_uint8 = (lut * 255).astype(np.uint8)  # convert to bytes for PG

        img_item.setLookupTable(lut_uint8)
        img_item.setLevels((0.0, 1.0))

        plot.addItem(img_item)

        rect = pg.QtCore.QRectF(-0.5, -0.5, W, H)
        img_item.setRect(rect)
        vb.setRange(rect, padding=0.0)

        plot.setTitle("Stimulation Area (Mask)")

        # Legend: only stimulation area
        if legend is not None:
            legend.clear()
            stim_area_item = pg.PlotDataItem(pen=pg.mkPen("yellow", width=2))
            legend.addItem(stim_area_item, "Stimulation Area")
            legend.setVisible(True)

    # ------------------ JUST STATS ------------------ #
    else:
        _display_roi_statistics_pg(
            plot, stimulated_roi_labels, non_stimulated_roi_labels
        )
        # No legend for pure text stats


def _display_roi_statistics_pg(
    plot: pg.PlotItem,
    stimulated_rois: list[int],
    non_stimulated_rois: list[int],
) -> None:
    """Display ROI counts as centered text (pyqtgraph)."""
    plot.clear()
    total = len(stimulated_rois) + len(non_stimulated_rois)
    text_lines = [
        "Stimulated Area / ROI Statistics",
        "",
        f"Total ROIs: {total}",
        f"Stimulated ROIs: {len(stimulated_rois)}",
        f"Non-Stimulated ROIs: {len(non_stimulated_rois)}",
    ]
    txt = "\n".join(text_lines)

    text_item = pg.TextItem(txt, anchor=(0.5, 0.5), color="w")
    plot.addItem(text_item)
    text_item.setPos(0, 0)
    plot.getViewBox().autoRange()
    plot.setTitle("Stimulated Area Visualization")
