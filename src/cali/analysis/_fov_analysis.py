"""FOV-level analysis functions for computing correlation and synchrony matrices.

This module provides functions to compute pairwise correlation and synchrony
matrices across all active ROIs in a FOV, as well as population-level burst
detection. These metrics are computed once during analysis and stored in the
FOVAnalysis table for efficient retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cali.analysis._util import (
    _compute_zero_lag_corr_matrix,
    _detect_population_bursts,
    _get_calcium_peaks_event_correlations_matrix,
    _get_calcium_peaks_event_synchrony,
    _get_spike_correlations_matrix,
    _get_spike_synchrony,
)
from cali.logger import cali_logger
from cali.sqlmodel._model import FOVAnalysis

if TYPE_CHECKING:
    from cali.sqlmodel._model import FOV, ROI, AnalysisSettings


def compute_fov_analysis(
    fov: FOV,
    analysis_settings: AnalysisSettings,
) -> FOVAnalysis | None:
    """Compute FOV-level correlation and synchrony analysis.

    This function calculates 6 pairwise metrics for all active ROIs in a FOV:

    DF/F and Deconvolved DF/F Calcium Traces
    -------------------
    1. Zero-lag Pearson correlation on ΔF/F traces
    2. Zero-lag Pearson correlation on deconvolved ΔF/F traces

    Calcium Peaks
    -------------
    3. Jitter synchrony on calcium peak events (± jitter window)
    4. Max-lag correlation on calcium peak events (within ± max_lag)

    Inferred Spikes
    ---------------
    5. Zero-lag Pearson correlation on binary spike trains
    6. Max-lag CCG-like correlation on spike trains (within ± max_lag)
    7. Jitter synchrony on spike trains (± jitter window)

    It requires that ROIs have traces and data_analysis attached
    (either via history or _new_* attributes).

    Parameters
    ----------
    fov : FOV
        FOV containing ROIs with traces and analysis data
    analysis_settings : AnalysisSettings
        Settings containing jitter_window and max_lag parameters

    Returns
    -------
    FOVAnalysis | None
        FOVAnalysis object with computed matrices, or None if insufficient data
    """
    # Collect active ROIs with their traces and analysis data
    active_rois: list[ROI] = []
    for roi in fov.rois:
        if not roi.active:
            continue
        active_rois.append(roi)

    if len(active_rois) < 2:
        cali_logger.info(
            f"FOV {fov.name}: Not enough active ROIs ({len(active_rois)}) "
            "for correlation/synchrony analysis. Need at least 2."
        )
        return None

    # Get traces and analysis data for active ROIs
    # Use _new_traces/_new_data_analysis if available (during extraction/analysis)
    # Otherwise fall back to traces_history/data_analysis_history
    roi_labels: list[int] = []
    dff_traces: list[np.ndarray] = []
    dec_dff_traces: list[np.ndarray] = []
    spike_trains: list[np.ndarray] = []
    peak_events_dict: dict[str, list[float]] = {}
    spike_data_dict: dict[str, list[float]] = {}

    for roi in active_rois:
        if roi.label_value is None:
            continue

        # Get traces - prefer _new_traces if available
        traces = None
        if hasattr(roi, "_new_traces") and roi._new_traces:
            traces = roi._new_traces[-1]  # Most recent
        elif roi.traces_history:
            traces = roi.traces_history[-1]

        if traces is None or traces.dec_dff is None:
            continue

        # Get analysis data - prefer _new_data_analysis if available
        data_analysis = None
        if hasattr(roi, "_new_data_analysis") and roi._new_data_analysis:
            data_analysis = roi._new_data_analysis[-1]
        elif roi.data_analysis_history:
            data_analysis = roi.data_analysis_history[-1]

        dff = np.asarray(traces.dff, dtype=float)
        if dff.ndim != 1 or dff.size == 0:
            continue

        dec_dff = np.asarray(traces.dec_dff, dtype=float)
        if dec_dff.ndim != 1 or dec_dff.size == 0:
            continue

        roi_labels.append(int(roi.label_value))
        dff_traces.append(dff)
        dec_dff_traces.append(dec_dff)

        # Build peak event binary arrays
        if data_analysis is not None and data_analysis.peaks_dec_dff is not None:
            # Create binary peak event array
            peak_indices = [int(p) for p in data_analysis.peaks_dec_dff]
            peak_array = np.zeros(len(dec_dff), dtype=float)
            for idx in peak_indices:
                if 0 <= idx < len(peak_array):
                    peak_array[idx] = 1.0
            peak_events_dict[str(roi.label_value)] = peak_array.tolist()

        # Build spike trains for inferred spikes
        if traces.inferred_spikes is not None:
            spikes = np.asarray(traces.inferred_spikes, dtype=float)
            spike_threshold = (
                data_analysis.inferred_spikes_threshold
                if data_analysis is not None
                else None
            )

            if spike_threshold is not None:
                # Threshold and binarize
                spikes[spikes <= spike_threshold] = 0.0
                spike_train = (spikes > 0.0).astype(float)
                if spike_train.sum() > 0:
                    spike_trains.append(spike_train)
                    spike_data_dict[str(roi.label_value)] = spike_train.tolist()

    if len(roi_labels) < 2:
        cali_logger.info(
            f"FOV {fov.name}: Not enough ROIs with valid traces "
            f"({len(roi_labels)}) for correlation analysis."
        )
        return None

    # Calcium trace metrics: ΔF/F and deconvolved ΔF/F
    # 1. Zero-lag correlation on deconvolved DF/F traces
    calcium_dff_corr_matrix = _compute_zero_lag_corr_matrix(dff_traces)

    # 2. Zero-lag correlation on deconvolved DF/F traces
    calcium_dec_dff_corr_matrix = _compute_zero_lag_corr_matrix(dec_dff_traces)

    # Convert milliseconds to frames using frame_rate
    frame_rate = analysis_settings.frame_rate  # frames per second

    # Helper function to convert ms to frames
    def ms_to_frames(ms: float) -> int:
        """Convert milliseconds to frames based on frame rate.

        Returns integer number of frames, minimum 0.

        Note: If ms is smaller than one frame period, this returns 0, which is
        intentional and mathematically sound:
        - For jitter_window=0: Only exact frame coincidence is counted
        - For max_lag=0: Only zero-lag correlation is computed (standard Pearson)

        This provides a graceful fallback for small time windows at high frame rates.
        """
        # ms / 1000 = seconds
        # seconds * fps = frames
        return max(0, int((ms / 1000.0) * frame_rate))

    # 3. Jitter synchrony on calcium peaks
    calcium_peaks_jitter_sync_matrix = None
    global_calcium_peaks_jitter_sync = None
    if len(peak_events_dict) >= 2:
        jitter_window_ms = analysis_settings.calcium_sync_jitter_window
        jitter_window_frames = ms_to_frames(jitter_window_ms)
        calcium_peaks_jitter_sync_matrix, _ = (
            _get_calcium_peaks_event_correlations_matrix(
                peak_events_dict,
                method="jitter_window",
                jitter_window=jitter_window_frames,
            )
        )
        if calcium_peaks_jitter_sync_matrix is not None:
            global_calcium_peaks_jitter_sync = _get_calcium_peaks_event_synchrony(
                calcium_peaks_jitter_sync_matrix
            )

    # 4. Max lag correlation on calcium peaks
    calcium_peaks_max_lag_corr_matrix = None
    global_calcium_peaks_max_lag_corr = None
    if len(peak_events_dict) >= 2:
        max_lag_ms = analysis_settings.calcium_peaks_max_lag
        max_lag_frames = ms_to_frames(max_lag_ms)
        (
            calcium_peaks_max_lag_corr_matrix,
            _,
        ) = _get_calcium_peaks_event_correlations_matrix(
            peak_events_dict,
            method="cross_correlation",
            max_lag=max_lag_frames,
        )
        if calcium_peaks_max_lag_corr_matrix is not None:
            global_calcium_peaks_max_lag_corr = _get_calcium_peaks_event_synchrony(
                calcium_peaks_max_lag_corr_matrix
            )

    # Compute spike metrics (3 measurements):
    # 4. Zero-lag correlation on spike trains
    spike_corr_matrix = None
    # 5. Max lag correlation on spikes
    spike_max_lag_corr_matrix = None
    spike_max_lag_values_matrix = None
    global_spike_max_lag_corr = None
    # 6. Jitter synchrony on spikes
    spike_jitter_sync_matrix = None
    global_spike_jitter_sync = None

    if len(spike_data_dict) >= 2:
        # 4. Zero-lag Pearson correlation on spike trains
        spike_corr_matrix = _compute_zero_lag_corr_matrix(spike_trains)

        # 5. Max lag correlation on spikes
        max_lag_ms = analysis_settings.spikes_sync_cross_corr_lag
        max_lag_frames = ms_to_frames(max_lag_ms)
        (
            spike_max_lag_corr_matrix,
            spike_max_lag_values_matrix,
        ) = _get_spike_correlations_matrix(
            spike_data_dict,
            method="cross_correlation",
            max_lag=max_lag_frames,
        )
        if spike_max_lag_corr_matrix is not None:
            global_spike_max_lag_corr = _get_spike_synchrony(spike_max_lag_corr_matrix)

        # 6. Jitter synchrony on spikes
        jitter_window_ms = analysis_settings.spikes_sync_jitter_window
        jitter_window_frames = ms_to_frames(jitter_window_ms)
        spike_jitter_sync_matrix, _ = _get_spike_correlations_matrix(
            spike_data_dict,
            method="jitter_window",
            jitter_window=jitter_window_frames,
        )
        if spike_jitter_sync_matrix is not None:
            global_spike_jitter_sync = _get_spike_synchrony(spike_jitter_sync_matrix)

    # --- Burst detection on population spike activity ---
    burst_count: int | None = None
    burst_avg_duration: float | None = None
    burst_avg_interval: float | None = None

    if len(spike_trains) >= 2:
        burst_count, burst_avg_duration, burst_avg_interval = _detect_population_bursts(
            spike_trains=spike_trains,
            frame_rate=analysis_settings.frame_rate,
            burst_threshold_percent=analysis_settings.burst_threshold,
            min_duration_ms=analysis_settings.burst_min_duration,
            gaussian_sigma_sec=analysis_settings.burst_gaussian_sigma,
        )

    # Create FOVAnalysis object with all measurements
    fov_analysis = FOVAnalysis(
        active_roi_labels=roi_labels,
        # Calcium metrics
        calcium_dff_correlation_matrix=(
            calcium_dff_corr_matrix.tolist()
            if calcium_dff_corr_matrix is not None
            else None
        ),
        calcium_dec_dff_corr_matrix=(
            calcium_dec_dff_corr_matrix.tolist()
            if calcium_dec_dff_corr_matrix is not None
            else None
        ),
        calcium_peaks_jitter_synchrony_matrix=(
            calcium_peaks_jitter_sync_matrix.tolist()
            if calcium_peaks_jitter_sync_matrix is not None
            else None
        ),
        global_calcium_peaks_jitter_synchrony=global_calcium_peaks_jitter_sync,
        calcium_peaks_max_lag_correlation_matrix=(
            calcium_peaks_max_lag_corr_matrix.tolist()
            if calcium_peaks_max_lag_corr_matrix is not None
            else None
        ),
        global_calcium_peaks_max_lag_correlation=global_calcium_peaks_max_lag_corr,
        # Spike metrics
        spike_correlation_matrix=(
            spike_corr_matrix.tolist() if spike_corr_matrix is not None else None
        ),
        spike_max_lag_correlation_matrix=(
            spike_max_lag_corr_matrix.tolist()
            if spike_max_lag_corr_matrix is not None
            else None
        ),
        global_spike_max_lag_correlation=global_spike_max_lag_corr,
        spike_max_lag_values_matrix=(
            spike_max_lag_values_matrix.tolist()
            if spike_max_lag_values_matrix is not None
            else None
        ),
        spike_jitter_synchrony_matrix=(
            spike_jitter_sync_matrix.tolist()
            if spike_jitter_sync_matrix is not None
            else None
        ),
        global_spike_jitter_synchrony=global_spike_jitter_sync,
        # Population burst metrics
        burst_count=burst_count,
        burst_avg_duration=burst_avg_duration,
        burst_avg_interval=burst_avg_interval,
    )

    return fov_analysis
