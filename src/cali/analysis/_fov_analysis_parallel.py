"""Parallelized FOV analysis using multiprocessing for CCG computation.

This module provides a parallelized version of compute_fov_analysis that uses
multiprocessing to speed up the expensive CCG (cross-correlogram) computation.

KEY DESIGN:
===========
1. Use multiprocessing.Pool for CCG pairs (avoids numba lock contention)
2. Create pool ONCE per FOV (not per operation)
3. Reuse worker processes for both CCG and jitter computations

WHEN TO USE:
============
- FOVs with many ROIs (>20)
- When CCG computation dominates total time (typically 95%+)
- NOT when running many FOVs in parallel threads (use sequential FOV processing)
"""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np

from cali.analysis._cluster_analysis import compute_cluster_analysis
from cali.analysis._fov_metrics import (
    _compute_baseline_corrected_ccg_numba,
    _compute_zero_lag_corr_matrix,
    _detect_calcium_population_bursts,
    _detect_spikes_population_bursts,
    _get_spike_synchrony,
    _jitter_window_synchrony_numba,
)
from cali.analysis._trace_analysis import (
    compute_rising_edges,
    threshold_spike_train,
)
from cali.logger import cali_logger
from cali.sqlmodel._model import FOVAnalysis

if TYPE_CHECKING:
    from cali.sqlmodel._model import FOV, AnalysisSettings


def _compute_ccg_for_pair(args: tuple) -> tuple[int, int, float, int, float]:
    """Worker function to compute CCG for a single ROI pair.

    Parameters
    ----------
    args : tuple
        (i, j, spike_i, spike_j, max_lag, n_shuffles)

    Returns
    -------
    tuple
        (i, j, max_ccg, best_lag, zscore)
    """
    i, j, spike_i, spike_j, max_lag, n_shuffles = args

    if np.sum(spike_i) == 0 or np.sum(spike_j) == 0:
        return (i, j, 0.0, 0, 0.0)

    lags, ccg_raw, baseline_mean, baseline_std = _compute_baseline_corrected_ccg_numba(
        spike_i, spike_j, max_lag, n_shuffles
    )

    max_idx = np.argmax(ccg_raw)
    max_value = float(ccg_raw[max_idx])
    best_lag = int(lags[max_idx])

    if baseline_std[max_idx] > 0:
        zscore = (ccg_raw[max_idx] - baseline_mean[max_idx]) / baseline_std[max_idx]
    else:
        zscore = 0.0

    return (i, j, max_value, best_lag, float(zscore))


def _compute_jitter_for_pair(args: tuple) -> tuple[int, int, float]:
    """Worker function to compute jitter synchrony for a single ROI pair.

    Parameters
    ----------
    args : tuple
        (i, j, spike_i, spike_j, jitter_window)

    Returns
    -------
    tuple
        (i, j, synchrony_value)
    """
    i, j, spike_i, spike_j, jitter_window = args

    if np.sum(spike_i) == 0 or np.sum(spike_j) == 0:
        return (i, j, 0.0)

    sync_value = float(_jitter_window_synchrony_numba(spike_i, spike_j, jitter_window))
    return (i, j, sync_value)


def _extract_fov_data(
    fov: FOV,
) -> tuple[
    list[int],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    """Extract data from FOV for analysis.

    Returns
    -------
    tuple
        (roi_labels, dff_traces, den_dff_traces, spike_trains,
         calcium_peak_events, spike_data_dict, spike_data_dict_rising_edges)
    """
    roi_labels: list[int] = []
    dff_traces: list[np.ndarray] = []
    den_dff_traces: list[np.ndarray] = []
    spike_trains: list[np.ndarray] = []
    calcium_peak_events: list[np.ndarray] = []
    spike_data_dict: dict[str, list[float]] = {}
    spike_data_dict_rising_edges: dict[str, list[float]] = {}

    for roi in fov.rois:
        if not roi.active or roi.label_value is None:
            continue

        # Get traces
        traces = None
        if hasattr(roi, "_new_traces") and roi._new_traces:
            traces = roi._new_traces[-1]
        elif roi.traces_history:
            traces = roi.traces_history[-1]

        if traces is None or traces.den_dff is None:
            continue

        # Get analysis data
        data_analysis = None
        if hasattr(roi, "_new_data_analysis") and roi._new_data_analysis:
            data_analysis = roi._new_data_analysis[-1]
        elif roi.data_analysis_history:
            data_analysis = roi.data_analysis_history[-1]

        dff = np.asarray(traces.dff, dtype=float)
        if dff.ndim != 1 or dff.size == 0:
            continue

        den_dff = np.asarray(traces.den_dff, dtype=float)
        if den_dff.ndim != 1 or den_dff.size == 0:
            continue

        roi_labels.append(int(roi.label_value))
        dff_traces.append(dff)
        den_dff_traces.append(den_dff)

        # Build peak event binary arrays for calcium burst detection
        if data_analysis is not None and data_analysis.peaks_den_dff is not None:
            peak_indices = [int(p) for p in data_analysis.peaks_den_dff]
            peak_array = np.zeros(len(den_dff), dtype=float)
            for idx in peak_indices:
                if 0 <= idx < len(peak_array):
                    peak_array[idx] = 1.0
            calcium_peak_events.append(peak_array)

        # Build spike data
        if traces.inferred_spikes is not None:
            spikes = np.asarray(traces.inferred_spikes, dtype=float)
            spike_threshold = (
                data_analysis.inferred_spikes_threshold
                if data_analysis is not None
                else None
            )

            if spike_threshold is not None:
                spike_train = threshold_spike_train(spikes, spike_threshold)
                spike_trains.append(spike_train)
                spike_data_dict[str(roi.label_value)] = spike_train.tolist()

                # Compute rising edges
                spike_train_rising_edges = compute_rising_edges(spike_train)
                spike_data_dict_rising_edges[str(roi.label_value)] = (
                    spike_train_rising_edges.tolist()
                )

    return (
        roi_labels,
        dff_traces,
        den_dff_traces,
        spike_trains,
        calcium_peak_events,
        spike_data_dict,
        spike_data_dict_rising_edges,
    )


def compute_fov_analysis_parallel(
    fov: FOV,
    analysis_settings: AnalysisSettings,
) -> FOVAnalysis | None:
    """Compute FOV analysis with parallel CCG computation.

    This is a drop-in replacement for compute_fov_analysis that uses
    multiprocessing to parallelize the spike CCG computation.

    Parameters
    ----------
    fov : FOV
        FOV containing ROIs with traces and analysis data
    analysis_settings : AnalysisSettings
        Settings containing analysis parameters (uses n_processes for worker count)

    Returns
    -------
    FOVAnalysis | None
        FOVAnalysis object with computed matrices, or None if insufficient data
    """
    # Extract data from FOV
    (
        roi_labels,
        dff_traces,
        den_dff_traces,
        spike_trains,
        calcium_peak_events,
        spike_data_dict,
        spike_data_dict_rising_edges,
    ) = _extract_fov_data(fov)

    if len(roi_labels) < 2:
        cali_logger.info(
            f"FOV {fov.name}: Not enough ROIs with valid traces "
            f"({len(roi_labels)}) for correlation analysis."
        )
        return None

    n_rois = len(roi_labels)
    n_pairs = n_rois * (n_rois - 1) // 2
    # Use parallel only if there are enough ROIs to justify overhead
    # (10 ROIs = 45 pairs, reasonable threshold for multiprocessing)
    use_parallel = n_rois >= 10 and len(spike_trains) >= 2

    # Get number of workers from settings
    n_workers = max(1, analysis_settings.n_processes)

    # Calcium trace correlations (fast, no parallelization needed)
    calcium_dff_corr_matrix = _compute_zero_lag_corr_matrix(dff_traces)
    calcium_den_dff_corr_matrix = _compute_zero_lag_corr_matrix(den_dff_traces)

    # Cluster analysis on denoised ΔF/F correlation matrix
    cluster_labels = None
    cluster_method_used = None
    cluster_n = None
    cluster_silhouette = None
    cluster_order = None

    if calcium_den_dff_corr_matrix is not None and len(roi_labels) >= 3:
        cluster_result = compute_cluster_analysis(
            corr_matrix=calcium_den_dff_corr_matrix,
            method=analysis_settings.cluster_method,
            n_clusters=analysis_settings.cluster_n_clusters,
            max_k=analysis_settings.cluster_max_k,
        )
        if cluster_result is not None:
            cluster_labels = cluster_result.labels
            cluster_method_used = analysis_settings.cluster_method
            cluster_n = cluster_result.n_clusters
            cluster_silhouette = cluster_result.silhouette_score
            cluster_order = cluster_result.order

    # Convert ms to frames
    frame_rate = analysis_settings.frame_rate

    def ms_to_frames(ms: float) -> int:
        return max(0, int((ms / 1000.0) * frame_rate))

    max_lag_frames = ms_to_frames(analysis_settings.spikes_sync_cross_corr_lag)
    jitter_window_frames = ms_to_frames(analysis_settings.spikes_sync_jitter_window)
    n_shuffles = analysis_settings.ccg_n_shuffles

    # Initialize matrices
    spike_max_lag_corr_matrix = None
    spike_max_lag_values_matrix = None
    global_spike_max_lag_corr = None
    spike_ccg_zscore_matrix = None
    spike_jitter_sync_matrix = None
    global_spike_jitter_sync = None
    # Rising edges
    spike_max_lag_corr_matrix_rising_edges = None
    spike_max_lag_values_matrix_rising_edges = None
    global_spike_max_lag_corr_rising_edges = None
    spike_ccg_zscore_matrix_rising_edges = None
    spike_jitter_sync_matrix_rising_edges = None
    global_spike_jitter_sync_rising_edges = None

    if len(spike_trains) >= 2:
        spike_trains_array = np.array(spike_trains, dtype=np.float32)
        pairs = [(i, j) for i in range(n_rois) for j in range(i + 1, n_rois)]

        if use_parallel:
            cali_logger.info(
                f"FOV {fov.name}: Parallel CCG ({n_rois} ROIs, {n_pairs} pairs, "
                f"{n_workers} workers)"
            )

            # Prepare arguments
            ccg_args = [
                (
                    i,
                    j,
                    spike_trains_array[i],
                    spike_trains_array[j],
                    max_lag_frames,
                    n_shuffles,
                )
                for i, j in pairs
            ]

            jitter_args = [
                (
                    i,
                    j,
                    spike_trains_array[i],
                    spike_trains_array[j],
                    jitter_window_frames,
                )
                for i, j in pairs
            ]

            # Use a single pool for all computations
            # 'spawn' is safer for numba but has overhead;
            # 'fork' is faster but can cause issues
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                # CCG computation
                ccg_results = pool.map(_compute_ccg_for_pair, ccg_args)

                # Jitter computation (reuse same pool)
                jitter_results = pool.map(_compute_jitter_for_pair, jitter_args)

            # Assemble CCG results
            spike_max_lag_corr_matrix = np.zeros((n_rois, n_rois))
            spike_max_lag_values_matrix = np.zeros((n_rois, n_rois), dtype=int)
            spike_ccg_zscore_matrix = np.zeros((n_rois, n_rois))

            np.fill_diagonal(spike_max_lag_corr_matrix, 1.0)
            np.fill_diagonal(spike_max_lag_values_matrix, 0)
            np.fill_diagonal(spike_ccg_zscore_matrix, np.inf)

            for i, j, max_ccg, best_lag, zscore in ccg_results:
                spike_max_lag_corr_matrix[i, j] = max_ccg
                spike_max_lag_corr_matrix[j, i] = max_ccg
                spike_max_lag_values_matrix[i, j] = best_lag
                spike_max_lag_values_matrix[j, i] = -best_lag
                spike_ccg_zscore_matrix[i, j] = zscore
                spike_ccg_zscore_matrix[j, i] = zscore

            global_spike_max_lag_corr = _get_spike_synchrony(spike_max_lag_corr_matrix)

            # Assemble jitter results
            spike_jitter_sync_matrix = np.zeros((n_rois, n_rois))
            np.fill_diagonal(spike_jitter_sync_matrix, 1.0)

            for i, j, sync_value in jitter_results:
                spike_jitter_sync_matrix[i, j] = sync_value
                spike_jitter_sync_matrix[j, i] = sync_value

            global_spike_jitter_sync = _get_spike_synchrony(spike_jitter_sync_matrix)

        else:
            # Sequential computation for small FOVs
            from cali.analysis._fov_metrics import _get_spike_correlations_matrix

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
                global_spike_max_lag_corr = _get_spike_synchrony(
                    spike_max_lag_corr_matrix
                )

            spike_jitter_sync_matrix, _, _ = _get_spike_correlations_matrix(
                spike_data_dict,
                method="jitter_window",
                jitter_window=jitter_window_frames,
            )
            if spike_jitter_sync_matrix is not None:
                global_spike_jitter_sync = _get_spike_synchrony(
                    spike_jitter_sync_matrix
                )

        # Rising edge analysis (if enabled)
        if (
            analysis_settings.enable_rising_edge_analysis
            and len(spike_data_dict_rising_edges) >= 2
        ):
            from cali.analysis._fov_metrics import _get_spike_correlations_matrix

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

            (
                spike_jitter_sync_matrix_rising_edges,
                _,
                _,
            ) = _get_spike_correlations_matrix(
                spike_data_dict_rising_edges,
                method="jitter_window",
                jitter_window=jitter_window_frames,
            )
            if spike_jitter_sync_matrix_rising_edges is not None:
                global_spike_jitter_sync_rising_edges = _get_spike_synchrony(
                    spike_jitter_sync_matrix_rising_edges
                )

    # Burst detection (fast, no parallelization needed)
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
            spike_population_activity_raw,
            spike_population_activity,
        ) = _detect_spikes_population_bursts(
            spike_trains=spike_trains,
            frame_rate=analysis_settings.frame_rate,
            burst_threshold_percent=analysis_settings.burst_threshold,
            min_duration_ms=analysis_settings.burst_min_duration,
            gaussian_sigma_sec=analysis_settings.burst_gaussian_sigma,
        )

    calcium_burst_count: int | None = None
    calcium_burst_avg_duration: float | None = None
    calcium_burst_avg_interval: float | None = None
    calcium_burst_starts: list[int] = []
    calcium_burst_ends: list[int] = []
    calcium_population_activity: np.ndarray | None = None
    calcium_population_activity_raw: np.ndarray | None = None

    if len(calcium_peak_events) >= 2:
        (
            calcium_burst_count,
            calcium_burst_avg_duration,
            calcium_burst_avg_interval,
            calcium_burst_starts,
            calcium_burst_ends,
            calcium_population_activity_raw,
            calcium_population_activity,
        ) = _detect_calcium_population_bursts(
            peak_events=calcium_peak_events,
            frame_rate=analysis_settings.frame_rate,
            burst_threshold_percent=analysis_settings.calcium_burst_threshold,
            min_duration_ms=analysis_settings.calcium_burst_min_duration,
            gaussian_sigma_sec=analysis_settings.calcium_burst_gaussian_sigma,
        )

    # Create FOVAnalysis object
    fov_analysis = FOVAnalysis(
        active_roi_labels=roi_labels,
        calcium_dff_correlation_matrix=(
            calcium_dff_corr_matrix.tolist()
            if calcium_dff_corr_matrix is not None
            else None
        ),
        calcium_den_dff_corr_matrix=(
            calcium_den_dff_corr_matrix.tolist()
            if calcium_den_dff_corr_matrix is not None
            else None
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
        spike_ccg_zscore_matrix=(
            spike_ccg_zscore_matrix.tolist()
            if spike_ccg_zscore_matrix is not None
            else None
        ),
        spike_jitter_synchrony_matrix=(
            spike_jitter_sync_matrix.tolist()
            if spike_jitter_sync_matrix is not None
            else None
        ),
        global_spike_jitter_synchrony=global_spike_jitter_sync,
        # Rising edges
        spike_max_lag_correlation_matrix_rising_edges=(
            spike_max_lag_corr_matrix_rising_edges.tolist()
            if spike_max_lag_corr_matrix_rising_edges is not None
            else None
        ),
        global_spike_max_lag_correlation_rising_edges=global_spike_max_lag_corr_rising_edges,
        spike_max_lag_values_matrix_rising_edges=(
            spike_max_lag_values_matrix_rising_edges.tolist()
            if spike_max_lag_values_matrix_rising_edges is not None
            else None
        ),
        spike_ccg_zscore_matrix_rising_edges=(
            spike_ccg_zscore_matrix_rising_edges.tolist()
            if spike_ccg_zscore_matrix_rising_edges is not None
            else None
        ),
        spike_jitter_synchrony_matrix_rising_edges=(
            spike_jitter_sync_matrix_rising_edges.tolist()
            if spike_jitter_sync_matrix_rising_edges is not None
            else None
        ),
        global_spike_jitter_synchrony_rising_edges=global_spike_jitter_sync_rising_edges,
        # Burst metrics
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
        # Cluster analysis results
        cluster_labels=cluster_labels,
        cluster_method=cluster_method_used,
        cluster_n_clusters=cluster_n,
        cluster_silhouette_score=cluster_silhouette,
        cluster_order=cluster_order,
    )

    return fov_analysis
