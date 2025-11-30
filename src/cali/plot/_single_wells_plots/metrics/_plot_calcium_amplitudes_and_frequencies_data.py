from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


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
    """Plot amplitude and/or frequency summary data by querying the database.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    run_id : int | None
        The run ID to filter by, None for latest (currently required)
    amp : bool
        Plot amplitude data (mean ± SEM of peaks_amplitudes_dec_dff)
    freq : bool
        Plot frequency data (dec_dff_frequency)
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    if run_id is None:
        cali_logger.warning("No run_id provided for amplitude/frequency plot.")
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
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
        ax.text(
            0.5,
            0.5,
            "No ROI analysis data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Plot each ROI's metrics
    for roi, data_analysis in roi_data:
        _plot_metrics(ax, roi, data_analysis, amp, freq)

    _set_graph_title_and_labels(ax, amp, freq)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi: ROI,
    data_analysis: DataAnalysis,
    amp: bool,
    freq: bool,
) -> None:
    """Plot amplitude and/or frequency summary for a single ROI."""
    # Amplitude vs frequency scatter (mean ± SEM vs frequency)
    if amp and freq:
        if (
            not data_analysis.peaks_amplitudes_dec_dff
            or data_analysis.dec_dff_frequency is None
        ):
            return

        amps = np.asarray(data_analysis.peaks_amplitudes_dec_dff, dtype=float)
        mean_amp = float(np.mean(amps))

        if amps.size > 1:
            std_amp = float(np.std(amps, ddof=1))
            sem_amp = std_amp / np.sqrt(amps.size)
        else:
            sem_amp = 0.0

        _plot_errorbars(
            ax,
            [float(data_analysis.dec_dff_frequency)],
            [mean_amp],
            [sem_amp],
            f"ROI {roi.label_value}",
            picker=5,
        )

    # Amplitude-only: per-ROI point at x = ROI label
    elif amp:
        if not data_analysis.peaks_amplitudes_dec_dff:
            return

        amps = np.asarray(data_analysis.peaks_amplitudes_dec_dff, dtype=float)
        mean_amp = float(np.mean(amps))

        if amps.size > 1:
            std_amp = float(np.std(amps, ddof=1))
            sem_amp = std_amp / np.sqrt(amps.size)
        else:
            sem_amp = 0.0

        _plot_errorbars(
            ax,
            [float(roi.label_value)],
            [mean_amp],
            [sem_amp],
            f"ROI {roi.label_value}",
            picker=5,
        )

        # Also show individual amplitudes as gray background points
        ax.scatter(
            [float(roi.label_value)] * amps.size,
            amps,
            alpha=0.5,
            s=30,
            color="lightgray",
            label=f"ROI {roi.label_value}",
            picker=True,
        )

    # Frequency-only: per-ROI point at x = ROI label
    elif freq:
        if data_analysis.dec_dff_frequency is None:
            return
        ax.plot(
            float(roi.label_value),
            float(data_analysis.dec_dff_frequency),
            "o",
            label=f"ROI {roi.label_value}",
            picker=5,
        )


def _plot_errorbars(
    ax: Axes,
    x: list[float],
    y: float | list[float],
    yerr: Any,
    label: str,
    picker: int | None = None,
) -> None:
    """Plot error bars."""
    errorbar = ax.errorbar(
        x,
        y,
        yerr=yerr,
        label=label,
        fmt="o",
        capsize=5,
        picker=picker,
    )
    if picker is None:
        return

    # Also enable picking on the marker artist (the mean point)
    if hasattr(errorbar, "lines") and errorbar.lines:
        errorbar.lines[0].set_picker(picker)
        errorbar.lines[0].set_label(label)


def _set_graph_title_and_labels(
    ax: Axes,
    amp: bool,
    freq: bool,
) -> None:
    """Set axis labels based on the plotted data."""
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

    ax.set_title(title)
    ax.set_ylabel(y_lbl)
    ax.set_xlabel(x_lbl)

    # For per-ROI plots, hide numeric x tick labels (visual clutter)
    if x_lbl == "ROIs":
        ax.set_xticks([])
        ax.set_xticklabels([])


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click(ax, widget, picker_tolerance=5)
