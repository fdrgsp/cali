"""FOV-level analysis functions for computing correlation and synchrony matrices.

This module provides functions to compute pairwise correlation and synchrony
matrices across all active ROIs in a FOV, as well as population-level burst
detection. These metrics are computed once during analysis and stored in the
FOVAnalysis table for efficient retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cali.analysis._fov_metrics import (
    _compute_zero_lag_corr_matrix,
    _detect_calcium_population_bursts,
    _detect_spikes_population_bursts,
    _get_spike_correlations_matrix,
    _get_spike_synchrony,
)
from cali.analysis._trace_analysis import (
    compute_rising_edges,
    threshold_spike_train,
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

    This function calculates 5 pairwise metrics for all active ROIs in a FOV:

    DF/F and Deconvolved DF/F Calcium Traces
    -------------------
    1. Zero-lag Pearson correlation on ΔF/F traces
    2. Zero-lag Pearson correlation on deconvolved ΔF/F traces

    Inferred Spikes
    ---------------
    3. Max-lag CCG-like correlation on binary spike trains (within ± max_lag)
    4. Jitter synchrony on binary spike trains (± jitter window)

    Burst Detection
    ---------------
    5. Additionally, population-level burst detection is performed on both the
    inferred spike trains and deconvolved ΔF/F traces, yielding metrics such as
    burst count, average duration, average interval, and population activity traces.

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
    spike_trains: list[np.ndarray] = []  # Binary (thresholded) for CCG/jitter/bursts
    peak_events_dict: dict[str, list[float]] = {}
    spike_data_dict: dict[str, list[float]] = {}
    spike_data_dict_rising_edges: dict[str, list[float]] = {}

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

        # Build spike data for inferred spikes
        if traces.inferred_spikes is not None:
            spikes = np.asarray(traces.inferred_spikes, dtype=float)

            # Create binary spike trains for CCG, jitter synchrony, and bursts
            spike_threshold = (
                data_analysis.inferred_spikes_threshold
                if data_analysis is not None
                else None
            )

            if spike_threshold is not None:
                # Threshold and binarize
                spike_train = threshold_spike_train(spikes, spike_threshold)
                # Always append spike train, even if sum == 0
                # This ensures spike matrices have same dimensions as active_roi_labels
                spike_trains.append(spike_train)
                spike_data_dict[str(roi.label_value)] = spike_train.tolist()

                # Compute rising edges for this spike train
                spike_train_rising_edges = compute_rising_edges(spike_train)
                spike_data_dict_rising_edges[str(roi.label_value)] = (
                    spike_train_rising_edges.tolist()
                )

    if len(roi_labels) < 2:
        cali_logger.info(
            f"FOV {fov.name}: Not enough ROIs with valid traces "
            f"({len(roi_labels)}) for correlation analysis."
        )
        return None

    # Calcium trace metrics: ΔF/F and deconvolved ΔF/F
    # 1. Zero-lag correlation on ΔF/F traces
    calcium_dff_corr_matrix = _compute_zero_lag_corr_matrix(dff_traces)

    # 2. Zero-lag correlation on deconvolved ΔF/F traces
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

    # 3. Max lag correlation on spikes (standard CCG with baseline correction)
    spike_max_lag_corr_matrix = None
    spike_max_lag_values_matrix = None
    global_spike_max_lag_corr = None
    spike_ccg_zscore_matrix = None
    # 3b. Max lag correlation on spikes (rising edges)
    spike_max_lag_corr_matrix_rising_edges = None
    spike_max_lag_values_matrix_rising_edges = None
    global_spike_max_lag_corr_rising_edges = None
    spike_ccg_zscore_matrix_rising_edges = None
    # 4. Jitter synchrony on spikes (thresholded binary)
    spike_jitter_sync_matrix = None
    global_spike_jitter_sync = None
    # 4b. Jitter synchrony on spikes (rising edges)
    spike_jitter_sync_matrix_rising_edges = None
    global_spike_jitter_sync_rising_edges = None

    if len(spike_data_dict) >= 2:
        # 3a. Max lag correlation on spikes (thresholded binary)
        # Using standard CCG methodology with:
        # - Per-trigger probability normalization (trigger_prob)
        # - Border correction for unbiased estimates at large lags
        # - Baseline correction using shift predictor
        max_lag_ms = analysis_settings.spikes_sync_cross_corr_lag
        max_lag_frames = ms_to_frames(max_lag_ms)
        n_shuffles = analysis_settings.ccg_n_shuffles
        (
            spike_max_lag_corr_matrix,
            spike_max_lag_values_matrix,
            spike_ccg_zscore_matrix,
        ) = _get_spike_correlations_matrix(
            spike_data_dict,
            method="cross_correlation",
            max_lag=max_lag_frames,
            n_shuffles=n_shuffles,
        )
        if spike_max_lag_corr_matrix is not None:
            global_spike_max_lag_corr = _get_spike_synchrony(spike_max_lag_corr_matrix)

        # 3b. Max lag correlation on spikes (thresholded rising edges)
        # Only compute if enabled (approximately doubles CCG computation time)
        if (
            analysis_settings.enable_rising_edge_analysis
            and len(spike_data_dict_rising_edges) >= 2
        ):
            (
                spike_max_lag_corr_matrix_rising_edges,
                spike_max_lag_values_matrix_rising_edges,
                spike_ccg_zscore_matrix_rising_edges,
            ) = _get_spike_correlations_matrix(
                spike_data_dict_rising_edges,
                method="cross_correlation",
                max_lag=max_lag_frames,
                n_shuffles=n_shuffles,
            )
            if spike_max_lag_corr_matrix_rising_edges is not None:
                global_spike_max_lag_corr_rising_edges = _get_spike_synchrony(
                    spike_max_lag_corr_matrix_rising_edges
                )

        # 4. Jitter synchrony on spikes (thresholded binary)
        jitter_window_ms = analysis_settings.spikes_sync_jitter_window
        jitter_window_frames = ms_to_frames(jitter_window_ms)
        spike_jitter_sync_matrix, _, _ = _get_spike_correlations_matrix(
            spike_data_dict,
            method="jitter_window",
            jitter_window=jitter_window_frames,
        )
        if spike_jitter_sync_matrix is not None:
            global_spike_jitter_sync = _get_spike_synchrony(spike_jitter_sync_matrix)

        # 4b. Jitter synchrony on spikes (thresholded rising edges)
        if (
            analysis_settings.enable_rising_edge_analysis
            and len(spike_data_dict_rising_edges) >= 2
        ):
            spike_jitter_sync_matrix_rising_edges, _, _ = (
                _get_spike_correlations_matrix(
                    spike_data_dict_rising_edges,
                    method="jitter_window",
                    jitter_window=jitter_window_frames,
                )
            )
            if spike_jitter_sync_matrix_rising_edges is not None:
                global_spike_jitter_sync_rising_edges = _get_spike_synchrony(
                    spike_jitter_sync_matrix_rising_edges
                )

    # 5. Burst detection on population spike activity
    spike_burst_count: int | None = None
    spike_burst_avg_duration: float | None = None
    spike_burst_avg_interval: float | None = None
    spike_burst_starts: list[int] = []
    spike_burst_ends: list[int] = []
    spike_population_activity: np.ndarray | None = None
    spike_population_activity_raw: np.ndarray | None = None

    if len(spike_trains) >= 2:
        (
            spike_burst_count,
            spike_burst_avg_duration,
            spike_burst_avg_interval,
            spike_burst_starts,
            spike_burst_ends,
            spike_population_activity,
            spike_population_activity_raw,
        ) = _detect_spikes_population_bursts(
            spike_trains=spike_trains,
            frame_rate=analysis_settings.frame_rate,
            burst_threshold_percent=analysis_settings.burst_threshold,
            min_duration_ms=analysis_settings.burst_min_duration,
            gaussian_sigma_sec=analysis_settings.burst_gaussian_sigma,
        )

    # Burst detection on population calcium activity
    calcium_burst_count: int | None = None
    calcium_burst_avg_duration: float | None = None
    calcium_burst_avg_interval: float | None = None
    calcium_burst_starts: list[int] = []
    calcium_burst_ends: list[int] = []
    calcium_population_activity: np.ndarray | None = None
    calcium_population_activity_raw: np.ndarray | None = None

    if len(dec_dff_traces) >= 2:
        (
            calcium_burst_count,
            calcium_burst_avg_duration,
            calcium_burst_avg_interval,
            calcium_burst_starts,
            calcium_burst_ends,
            calcium_population_activity,
            calcium_population_activity_raw,
        ) = _detect_calcium_population_bursts(
            dec_dff_traces=dec_dff_traces,
            frame_rate=analysis_settings.frame_rate,
            burst_threshold_percent=analysis_settings.calcium_burst_threshold,
            min_duration_ms=analysis_settings.calcium_burst_min_duration,
            gaussian_sigma_sec=analysis_settings.calcium_burst_gaussian_sigma,
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
        # Spike metrics
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
        spike_max_lag_correlation_matrix_rising_edges=(
            spike_max_lag_corr_matrix_rising_edges.tolist()
            if spike_max_lag_corr_matrix_rising_edges is not None
            else None
        ),
        global_spike_max_lag_correlation_rising_edges=(
            global_spike_max_lag_corr_rising_edges
        ),
        spike_max_lag_values_matrix_rising_edges=(
            spike_max_lag_values_matrix_rising_edges.tolist()
            if spike_max_lag_values_matrix_rising_edges is not None
            else None
        ),
        # Z-score matrices for CCG significance (baseline-corrected)
        spike_ccg_zscore_matrix=(
            spike_ccg_zscore_matrix.tolist()
            if spike_ccg_zscore_matrix is not None
            else None
        ),
        spike_ccg_zscore_matrix_rising_edges=(
            spike_ccg_zscore_matrix_rising_edges.tolist()
            if spike_ccg_zscore_matrix_rising_edges is not None
            else None
        ),
        spike_jitter_synchrony_matrix=(
            spike_jitter_sync_matrix.tolist()
            if spike_jitter_sync_matrix is not None
            else None
        ),
        global_spike_jitter_synchrony=global_spike_jitter_sync,
        spike_jitter_synchrony_matrix_rising_edges=(
            spike_jitter_sync_matrix_rising_edges.tolist()
            if spike_jitter_sync_matrix_rising_edges is not None
            else None
        ),
        global_spike_jitter_synchrony_rising_edges=(
            global_spike_jitter_sync_rising_edges
        ),
        # Population burst metrics (spike-based)
        spike_burst_count=spike_burst_count,
        spike_burst_avg_duration=spike_burst_avg_duration,
        spike_burst_avg_interval=spike_burst_avg_interval,
        spike_burst_starts=spike_burst_starts if spike_burst_starts else None,
        spike_burst_ends=spike_burst_ends if spike_burst_ends else None,
        spike_population_activity=(
            spike_population_activity.tolist()
            if spike_population_activity is not None
            else None
        ),
        spike_population_activity_raw=(
            spike_population_activity_raw.tolist()
            if spike_population_activity_raw is not None
            else None
        ),
        # Population burst metrics (calcium-based)
        calcium_burst_count=calcium_burst_count,
        calcium_burst_avg_duration=calcium_burst_avg_duration,
        calcium_burst_avg_interval=calcium_burst_avg_interval,
        calcium_burst_starts=calcium_burst_starts if calcium_burst_starts else None,
        calcium_burst_ends=calcium_burst_ends if calcium_burst_ends else None,
        calcium_population_activity=(
            calcium_population_activity.tolist()
            if calcium_population_activity is not None
            else None
        ),
        calcium_population_activity_raw=(
            calcium_population_activity_raw.tolist()
            if calcium_population_activity_raw is not None
            else None
        ),
    )

    return fov_analysis
