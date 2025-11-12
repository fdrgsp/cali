from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import mplcursors
import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, create_engine, select

from cali.sqlmodel._model import FOV, ROI

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _plot_amplitude_and_frequency_data(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    rois: list[int] | None = None,
    amp: bool = False,
    freq: bool = False,
) -> None:
    """Plot amplitude and frequency data by querying database directly.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    db_path : str | Path
        Path to the SQLite database
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    amp : bool
        Plot amplitude data
    freq : bool
        Plot frequency data
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Query database for ROI data
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with Session(engine) as session:
        # Build query to get ROIs for this FOV with eager loading of related data
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .options(
                selectinload(ROI.data_analysis),  # type: ignore
            )
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    engine.dispose(close=True)

    # Plot the data
    for roi in roi_models:
        if roi.data_analysis is None:
            continue
        _plot_metrics(ax, roi, amp, freq)

    _set_graph_title_and_labels(ax, amp, freq)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi: ROI,
    amp: bool,
    freq: bool,
) -> None:
    """Plot amplitude or frequency for a single ROI."""
    if roi.data_analysis is None:
        return

    if amp and freq:
        if (not roi.data_analysis.peaks_amplitudes_dec_dff or
            roi.data_analysis.dec_dff_frequency is None):
            return
        mean_amp = cast("float", np.mean(roi.data_analysis.peaks_amplitudes_dec_dff))
        std_amp = np.std(roi.data_analysis.peaks_amplitudes_dec_dff, ddof=1)  # sample std
        sem_amp = std_amp / np.sqrt(len(roi.data_analysis.peaks_amplitudes_dec_dff))
        _plot_errorbars(
            ax, [roi.data_analysis.dec_dff_frequency], [mean_amp], [sem_amp],
            f"ROI {roi.label_value}"
        )
    elif amp:
        if not roi.data_analysis.peaks_amplitudes_dec_dff:
            return

        # plot mean amplitude +- sem of each ROI
        mean_amp = cast("float", np.mean(roi.data_analysis.peaks_amplitudes_dec_dff))
        std_amp = np.std(roi.data_analysis.peaks_amplitudes_dec_dff, ddof=1)  # sample std
        sem_amp = std_amp / np.sqrt(len(roi.data_analysis.peaks_amplitudes_dec_dff))
        _plot_errorbars(
            ax, [roi.label_value], [mean_amp], [sem_amp], f"ROI {roi.label_value}"
        )
        ax.scatter(
            [roi.label_value] * len(roi.data_analysis.peaks_amplitudes_dec_dff),
            roi.data_analysis.peaks_amplitudes_dec_dff,
            alpha=0.5,
            s=30,
            color="lightgray",
            label=f"ROI {roi.label_value}",
        )
    elif freq:
        if roi.data_analysis.dec_dff_frequency is None:
            return
        ax.plot(
            roi.label_value, roi.data_analysis.dec_dff_frequency, "o",
            label=f"ROI {roi.label_value}"
        )


def _plot_errorbars(
    ax: Axes, x: list[float], y: float | list[float], yerr: Any, label: str
) -> None:
    """Plot error bars graph."""
    ax.errorbar(x, y, yerr=yerr, label=label, fmt="o", capsize=5)


def _set_graph_title_and_labels(
    ax: Axes,
    amp: bool,
    freq: bool,
) -> None:
    """Set axis labels based on the plotted data."""
    title = x_lbl = y_lbl = ""
    if amp and freq:
        title = (
            "ROIs Mean Calcium Peaks Amplitude ± SEM vs Frequency (Deconvolved ΔF/F)"
        )
        x_lbl = "Frequency (Hz)"
        y_lbl = "Amplitude"
    elif amp:
        title = "Calcium Peaks Mean Amplitude ± SEM (Deconvolved ΔF/F)"
        x_lbl = "ROIs"
        y_lbl = "Amplitude"
    elif freq:
        title = "Calcium Peaks Frequency (Deconvolved ΔF/F)"
        x_lbl = "ROIs"
        y_lbl = "Frequency (Hz)"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)
    ax.set_xlabel(x_lbl)
    if x_lbl == "ROIs":
        ax.set_xticks([])
        ax.set_xticklabels([])


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for ROI traces, not for peaks or other elements
        if label and "ROI" in label and not label.startswith("_"):
            # Get the data point coordinates
            x, y = sel.target

            # Create hover text with ROI and value information
            roi = cast("str", label.split(" ")[1])

            # Determine what type of plot this is based on axis labels
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()

            if "Amplitude" in x_label and "Frequency" in y_label:
                # Amplitude vs Frequency plot
                hover_text = f"{label}\nAmp: {x:.3f}\nFreq: {y:.3f} Hz"
            elif "Amplitude" in y_label:
                # Amplitude plot
                hover_text = f"{label}\nAmp: {y:.3f}"
            elif "Frequency" in y_label:
                # Frequency plot
                hover_text = f"{label}\nFreq: {y:.3f} Hz"
            else:
                # Fallback to just ROI label
                hover_text = label

            sel.annotation.set(text=hover_text, fontsize=8, color="black")

            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
