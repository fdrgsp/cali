import re
from typing import TYPE_CHECKING, Callable

import numpy as np
import pyqtgraph as pg
from sqlalchemy.engine import Engine
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from cali._constants import MAX_FRAMES_AFTER_STIMULATION, MWCM
from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces
from cali.sqlmodel._util import ROIData

if TYPE_CHECKING:
    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


def equation_from_str(equation: str) -> Callable | None:
    """Parse various equation formats and return a callable function.

    Supported formats:
    - Linear: y = m*x + q  (e.g. "y = 2*x + 3")
    - Quadratic: y = a*x^2 + b*x + c  (e.g. "y = 0.5*x^2 + 2*x + 1")
    - Exponential: y = a*exp(b*x) + c  (e.g. "y = 2*exp(0.1*x) + 1")
    - Power: y = a*x^b + c  (e.g. "y = 2*x^0.5 + 1")
    - Logarithmic: y = a*log(x) + b  (e.g. "y = 2*log(x) + 1")
    """
    if not equation:
        return None

    # Remove all whitespace for easier parsing
    eq = equation.replace(" ", "").lower()

    try:
        if linear_match := re.match(r"y=([+-]?\d*\.?\d+)\*x([+-]\d*\.?\d+)", eq):
            m = float(linear_match[1])
            q = float(linear_match[2])
            return lambda x: m * x + q

        if quad_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*x\^2([+-]\d*\.?\d+)\*x([+-]\d*\.?\d+)", eq
        ):
            a = float(quad_match[1])
            b = float(quad_match[2])
            c = float(quad_match[3])
            return lambda x: a * x**2 + b * x + c

        if exp_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*exp\(([+-]?\d*\.?\d+)\*x\)([+-]\d*\.?\d+)",
            eq,
        ):
            a = float(exp_match[1])
            b = float(exp_match[2])
            c = float(exp_match[3])
            return lambda x: a * np.exp(b * x) + c

        if power_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*x\^([+-]?\d*\.?\d+)([+-]\d*\.?\d+)", eq
        ):
            a = float(power_match[1])
            b = float(power_match[2])
            c = float(power_match[3])
            return lambda x: a * (x**b) + c

        if log_match := re.match(r"y=([+-]?\d*\.?\d+)\*log\(x\)([+-]\d*\.?\d+)", eq):
            a = float(log_match[1])
            b = float(log_match[2])
            return lambda x: a * np.log(x) + b

        # If no pattern matches, show error
        msg = (
            "Invalid equation format! Using values from the metadata.\n"
            "Only Linear, Quadratic, Exponential, Power, and Logarithmic equations "
            "are supported."
        )
        cali_logger.error(msg)
        return None

    except ValueError as e:
        msg = (
            f"Error parsing equation coefficients: {e}\nUsing values from the metadata."
        )
        cali_logger.error(msg)
        return None


def _get_traces_for_run(roi: ROI, run_id: int | None) -> "Traces | None":
    """Get the Traces object for a specific run from the ROI's traces_history."""
    if not roi.traces_history:
        return None
    if run_id is None:
        return roi.traces_history[0] if roi.traces_history else None
    # First try to find exact match
    for trace in roi.traces_history:
        if trace.analysis_result_id == run_id:
            return trace
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return roi.traces_history[0] if roi.traces_history else None


def _get_data_analysis_for_run(roi: ROI, run_id: int | None) -> "DataAnalysis | None":
    """Get DataAnalysis for a specific run from ROI's data_analysis_history."""
    if not roi.data_analysis_history:
        return None
    if run_id is None:
        return roi.data_analysis_history[0] if roi.data_analysis_history else None
    # First try to find exact match
    for analysis in roi.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return roi.data_analysis_history[0] if roi.data_analysis_history else None


def get_stimulated_amplitudes_from_roi_data(
    roi_data: ROIData,
    led_power_equation: Callable | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Get stimulated and non-stimulated amplitudes from ROIData on-demand.

    Args:
        roi_data: ROIData object containing the necessary data
        led_power_equation: Optional function to convert power percentage to mW/cm²

    Returns
    -------
        Tuple of (amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks)
    """
    if (
        not roi_data.evoked_experiment
        or roi_data.den_dff is None
        or roi_data.peaks_den_dff is None
        or roi_data.stimulations_frames_and_powers is None
    ):
        return {}, {}

    return separate_stimulated_vs_non_stimulated_peaks(
        den_dff=np.array(roi_data.den_dff),
        peaks_den_dff=np.array(roi_data.peaks_den_dff),
        pulse_on_frames_and_powers=roi_data.stimulations_frames_and_powers,
        is_roi_stimulated=roi_data.stimulated,
        led_pulse_duration=roi_data.led_pulse_duration or "unknown",
        led_power_equation=led_power_equation,
    )


def separate_stimulated_vs_non_stimulated_peaks(
    den_dff: np.ndarray,
    peaks_den_dff: np.ndarray,
    pulse_on_frames_and_powers: dict[str, int],
    is_roi_stimulated: bool,
    led_pulse_duration: str = "unknown",
    led_power_equation: Callable | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Separate peak amplitudes into stimulated and non-stimulated categories.

    Args:
        den_dff: Denoised dF/F signal
        peaks_den_dff: Array of peak indices
        pulse_on_frames_and_powers: Dict mapping frame numbers to power values
        is_roi_stimulated: Whether this ROI is in a stimulated area
        led_pulse_duration: Duration of LED pulse (for labeling)
        led_power_equation: Optional function to convert power percentage to mW/cm²

    Returns
    -------
        Tuple of (amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks)
        Each is a dict mapping power_duration strings to lists of amplitudes
    """
    import bisect

    amplitudes_stimulated_peaks: dict[str, list[float]] = {}
    amplitudes_non_stimulated_peaks: dict[str, list[float]] = {}

    sorted_peaks_den_dff = sorted(peaks_den_dff)

    for frame, power in pulse_on_frames_and_powers.items():
        stim_frame = int(frame)
        # Find index of first peak >= stim_frame
        i = bisect.bisect_left(sorted_peaks_den_dff, stim_frame)

        # Check if index is valid
        if i >= len(sorted_peaks_den_dff):
            continue

        peak_idx = sorted_peaks_den_dff[i]

        # Check if peak is within stimulation window
        if (
            peak_idx >= stim_frame
            and peak_idx <= stim_frame + MAX_FRAMES_AFTER_STIMULATION
        ):
            amplitude = float(den_dff[peak_idx])

            # Format power value
            if led_power_equation is not None:
                power_val = led_power_equation(power)
                power_str = f"{power_val:.3f}{MWCM}"
            else:
                power_str = f"{power}%"

            # Create column key
            col = f"{power_str}_{led_pulse_duration}"

            # Categorize based on stimulation status
            if is_roi_stimulated:
                amplitudes_stimulated_peaks.setdefault(col, []).append(amplitude)
            else:
                amplitudes_non_stimulated_peaks.setdefault(col, []).append(amplitude)

    return amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks


def _get_spikes_over_threshold(
    engine: Engine, fov_name: str, roi_id: int, raw: bool = False
) -> list[float] | None:
    """Get spikes over threshold from ROI data."""
    with Session(engine) as session:
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.id) == roi_id)
            .options(
                selectinload(ROI.data_analysis_history),
                selectinload(ROI.traces_history),
            )
        )
        roi = session.exec(stmt).first()

    if roi is None or not roi.data_analysis_history or not roi.traces_history:
        return None

    # Use the first entry
    da = roi.data_analysis_history[0]
    trace = roi.traces_history[0]

    inferred_spikes = trace.inferred_spikes
    inferred_spikes_threshold = da.inferred_spikes_threshold

    if inferred_spikes is None or inferred_spikes_threshold is None:
        return None

    if raw:
        # Return raw inferred spikes
        return inferred_spikes  # type: ignore[no-any-return]

    spikes_thresholded = []
    for spike in inferred_spikes:
        if spike > inferred_spikes_threshold:
            spikes_thresholded.append(spike)
        else:
            spikes_thresholded.append(0.0)
    return spikes_thresholded


def disconnect_hover_handlers(plot: pg.PlotItem) -> None:
    """Disconnect hover and click handlers from previous plots to prevent conflicts."""
    scene = plot.scene()

    # Hover handlers (connected to sigMouseMoved)
    hover_handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_corr_hover_handler",
        "spike_ccorr_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in hover_handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # Click handlers (connected to sigMouseClicked)
    click_handler_names = [
        "amp_raster_click_handler",
        "intensity_heatmap_click_handler",
        "spike_raster_click_handler",
        "spike_intensity_heatmap_click_handler",
        "neuropil_click_handler",
        "peaks_amp_click_handler",
        "raster_click_handler",
        "connectivity_click_handler",
        "connectivity_bg_click_handler",
        "spike_maxlag_click_handler",
        "spike_maxlag_values_click_handler",
        "spike_sync_click_handler",
        "spike_ccorr_click_handler",
        "dff_corr_click_handler",
        "evoked_click_handler",
        "cell_size_click_handler",
    ]
    for handler_name in click_handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseClicked.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)


def add_colorbar_to_widget(
    widget: "_SingleWellGraphWidget",
    vmin: float,
    vmax: float,
    label: str = "Synchrony",
    colormap: str = "viridis",
) -> None:
    """Add a ColorBarItem to the widget layout."""
    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Create ColorBarItem
    widget.colorbar = pg.ColorBarItem(
        values=(vmin, vmax),
        colorMap=pg.colormap.get(colormap),
        width=15,
        label=label,
        interactive=False,
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)
