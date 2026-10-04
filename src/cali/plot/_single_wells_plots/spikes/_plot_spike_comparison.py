"""Compare aligned OASIS/CASCADE outputs without equating their amplitudes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg  # type: ignore[import-untyped, unused-ignore]

from cali.plot._spike_comparison import get_spike_comparisons
from cali.plot._util import disconnect_hover_handlers

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
    from cali.sqlmodel._spike_settings import SpikeMethod


def plot_spike_comparison(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    *,
    onsets: bool = False,
) -> None:
    """Render paired normalized traces or separately labelled threshold onsets."""
    plot = widget.plot_item
    plot.clear()
    disconnect_hover_handlers(plot)
    if widget.colorbar is not None:
        plot.layout.removeItem(widget.colorbar)
        widget.colorbar = None
    legend = widget.legend
    if legend is not None:
        legend.clear()
        legend.setVisible(False)
    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    vb.invertY(onsets)
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.enableAutoRange(x=True, y=True)
    plot.getAxis("bottom").setTicks(None)
    plot.getAxis("bottom").setStyle(showValues=True)
    plot.setLabel("bottom", "Retained Frames (0-based)")
    plot.setLabel(
        "left", "ROI / Method" if onsets else "Independently Normalized Amplitude"
    )
    plot.getAxis("left").setStyle(showValues=onsets)
    plot.getAxis("left").setTicks([])
    title = "OASIS / CASCADE Comparison — " + (
        "Threshold Onsets (Common Valid Interval)"
        if onsets
        else "Normalized Traces (Common Valid Interval)"
    )
    plot.setTitle(title)
    if run_id is None:
        plot.setTitle(title + " (Select one run)")
        return
    pairs = get_spike_comparisons(
        engine, fov_name, run_id, rois, require_threshold=onsets
    )
    if not pairs:
        plot.setTitle(title + " (No aligned dual-method data)")
        return
    methods: tuple[SpikeMethod, ...] = ("oasis", "cascade")
    colors = {"oasis": "#377eb8", "cascade": "#e66101"}
    ticks = []
    for index, pair in enumerate(pairs):
        for method_index, method in enumerate(methods):
            label = (
                "OASIS Rising Edges"
                if method == "oasis"
                else "CASCADE Threshold Excursion Starts"
            )
            if onsets:
                frames = pair.event_frames(method)
                row = index * 2 + method_index
                ticks.append((row, f"ROI {pair.roi_label} / {label}"))
                item = pg.ScatterPlotItem(
                    x=frames,
                    y=np.full(len(frames), row),
                    pen=None,
                    brush=pg.mkBrush(colors[method]),
                    size=6,
                )
                plot.addItem(item)
            else:
                item = plot.plot(
                    np.arange(pair.valid_start, pair.valid_stop),
                    pair.normalized_values(method) + index * 1.2,
                    pen=pg.mkPen(colors[method], width=2),
                )
                item.setCurveClickable(True, 8)
                item.sigClicked.connect(
                    lambda _curve, _event, label=pair.roi_label: (
                        widget.roiSelected.emit(str(label))
                    )
                )
            item.setProperty("roi_label", str(pair.roi_label))
            item.setProperty("spike_method", method)
            item.setToolTip(
                f"ROI {pair.roi_label}; {method.upper()}; common retained interval "
                f"[{pair.valid_start}, {pair.valid_stop}); amplitudes "
                + ("a.u." if method == "oasis" else "expected spikes/frame")
                + (
                    "; left boundary censored"
                    if onsets
                    else "; independent peak scaling"
                )
            )
    if onsets:
        plot.getAxis("left").setTicks([ticks])
    if legend is not None:
        for method in methods:
            legend.addItem(
                pg.PlotDataItem(pen=pg.mkPen(colors[method], width=2)), method.upper()
            )
        legend.setVisible(True)
