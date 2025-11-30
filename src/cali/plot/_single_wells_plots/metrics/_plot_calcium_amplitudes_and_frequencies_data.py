from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_hover
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
    # First try to find exact match
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return (
        roi_model.data_analysis_history[0] if roi_model.data_analysis_history else None
    )


def _plot_amplitude_and_frequency_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    amp: bool = False,
    freq: bool = False,
) -> None:
    """Plot amplitude and frequency data by querying database directly.

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
        The run ID to filter by, None for latest
    amp : bool
        Plot amplitude data
    freq : bool
        Plot frequency data
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Query database for ROI data
    with Session(engine) as session:
        roi_data = []  # List of (ROI, DataAnalysis)

        if run_id is None:
            cali_logger.warning("No run_id provided for IEI plot.")
            return

        # Optimized query
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

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()
        roi_data = results

    # Plot the data
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
    """Plot amplitude or frequency for a single ROI."""
    if amp and freq:
        if (
            not data_analysis.peaks_amplitudes_dec_dff
            or data_analysis.dec_dff_frequency is None
        ):
            return
        mean_amp = cast("float", np.mean(data_analysis.peaks_amplitudes_dec_dff))

        # Only calculate SEM if we have more than one data point
        if len(data_analysis.peaks_amplitudes_dec_dff) > 1:
            std_amp = np.std(data_analysis.peaks_amplitudes_dec_dff, ddof=1)
            sem_amp = std_amp / np.sqrt(len(data_analysis.peaks_amplitudes_dec_dff))
        else:
            sem_amp = 0  # No error bars for single point

        _plot_errorbars(
            ax,
            [data_analysis.dec_dff_frequency],
            [mean_amp],
            [sem_amp],
            f"ROI {roi.label_value}",
        )
    elif amp:
        if not data_analysis.peaks_amplitudes_dec_dff:
            return

        # plot mean amplitude +- sem of each ROI
        mean_amp = cast("float", np.mean(data_analysis.peaks_amplitudes_dec_dff))

        # Only calculate SEM if we have more than one data point
        if len(data_analysis.peaks_amplitudes_dec_dff) > 1:
            std_amp = np.std(data_analysis.peaks_amplitudes_dec_dff, ddof=1)
            sem_amp = std_amp / np.sqrt(len(data_analysis.peaks_amplitudes_dec_dff))
        else:
            sem_amp = 0  # No error bars for single point

        _plot_errorbars(
            ax, [roi.label_value], [mean_amp], [sem_amp], f"ROI {roi.label_value}"
        )
        ax.scatter(
            [roi.label_value] * len(data_analysis.peaks_amplitudes_dec_dff),
            data_analysis.peaks_amplitudes_dec_dff,
            alpha=0.5,
            s=30,
            color="lightgray",
            label=f"ROI {roi.label_value}",  # Add label so scatter is pickable
            picker=True,  # Enable picking on scatter
        )
    elif freq:
        if data_analysis.dec_dff_frequency is None:
            return
        ax.plot(
            roi.label_value,
            data_analysis.dec_dff_frequency,
            "o",
            label=f"ROI {roi.label_value}",
            picker=5,  # Enable picking
        )


def _plot_errorbars(
    ax: Axes, x: list[float], y: float | list[float], yerr: Any, label: str
) -> None:
    """Plot error bars graph."""
    errorbar = ax.errorbar(x, y, yerr=yerr, label=label, fmt="o", capsize=5, picker=5)
    # Also enable picking on the marker artist (the mean point)
    if hasattr(errorbar, "lines") and len(errorbar.lines) > 0:
        errorbar.lines[0].set_picker(5)


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
    """Add hover functionality using efficient pick events."""
    setup_pick_hover(ax, widget, picker_tolerance=5)
