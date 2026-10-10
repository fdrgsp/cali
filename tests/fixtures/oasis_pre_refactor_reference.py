"""Frozen pre-refactor OASIS ROI calculation for exact native-platform parity.

Copied from src/cali/extraction/_extraction_runner.py at
8340eaf9ef2e398bb4d3e3bcafb5289841ecea06, before the batch-interface refactor.
Original method SHA256: 0b1cde876f9f67be3cdc659bd324997e67d4c32c44c4d6871df4a8334756e14c
The computation is unchanged. Plain attribute containers replace the old database
models so their legacy fields survive the current method-qualified schema.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
from oasis.functions import GetSn, deconvolve, estimate_parameters

from cali.analysis._fov_metrics import get_overlap_roi_with_stimulated_area
from cali.analysis._trace_analysis import compute_inferred_spike_threshold
from cali.extraction._util import calculate_dff
from cali.logger import cali_logger
from cali.util._util import _NUMBA_LOCK

if TYPE_CHECKING:
    from cali.sqlmodel import AnalysisSettings, ExtractionSettings

Traces = SimpleNamespace
DataAnalysis = SimpleNamespace


class LegacyOasisReference:
    """Execute the original inline ROI path without database schema dependencies."""

    def _check_for_abort_requested(self) -> bool:
        return False

    def _process_roi_trace(
        self,
        data: np.ndarray,
        meta: list[dict],
        fov_name: str,
        extraction_settings: ExtractionSettings,
        analysis_settings: AnalysisSettings | None,
        label_value: int,
        label_mask: np.ndarray,
        tot_time_sec: float,
        elapsed_time_list: list[float],
        x_unit: str,
        neuropil_mask: np.ndarray | None = None,
        neuropil_correction_factor: float | None = None,
    ) -> tuple[Traces, DataAnalysis | None, bool, bool, float, str] | None:
        """Process individual ROI trace and return trace data.

        Parameters
        ----------
        data : np.ndarray
            Imaging data array (time, height, width)
        meta : list[dict]
            Metadata for the imaging data
        fov_name : str
            Name of the field of view
        extraction_settings : ExtractionSettings
            Settings for extraction (neuropil, dff_window, decay_constant)
        analysis_settings : AnalysisSettings | None
            Settings for analysis (peak detection, thresholds). If provided,
            peak detection and analysis will be performed.
        label_value : int
            ROI label value
        label_mask : np.ndarray
            Boolean mask for the ROI
        tot_time_sec : float
            Total recording time in seconds
        elapsed_time_list : list[float]
            List of elapsed times for each frame
        x_unit : str
            Unit for x-axis (e.g., "ms")
        neuropil_mask : np.ndarray | None
            Neuropil mask for correction
        neuropil_correction_factor : float | None
            Factor for neuropil correction

        Returns
        -------
        tuple[Traces, DataAnalysis | None, bool, bool, float, str] | None
            Tuple of (Traces, DataAnalysis | None, active, stimulated,
            roi_size, roi_size_units) ready to add to an existing ROI,
            or None if processing fails or ROI should be excluded.
            DataAnalysis will be None if run_analysis=False.
        """
        # Early exit if cancellation is requested
        if self._check_for_abort_requested():
            return None

        # get the data for the current label
        masked_data = data[:, label_mask]

        # get the size of the roi in µm or px if µm is not available
        roi_size_pixel = masked_data.shape[1]  # area
        # Try to get pixel size from settings
        px_size = extraction_settings.pixel_size
        # Convert to µm² if pixel size is available, otherwise use pixels
        roi_size = roi_size_pixel * (px_size**2) if px_size else roi_size_pixel
        roi_size_units = "µm²" if px_size is not None else "pixels"

        # Check for cancellation before DFF calculation
        if self._check_for_abort_requested():
            return None

        # compute the mean for each frame
        roi_trace_uncorrected: np.ndarray = masked_data.mean(axis=1)

        # Apply neuropil correction if enabled
        neuropil_trace = None
        roi_trace = roi_trace_uncorrected.copy()  # Start with uncorrected trace
        if neuropil_mask is not None and neuropil_correction_factor is not None:
            neuropil_masked_data = data[:, neuropil_mask]
            if neuropil_masked_data.shape[1] > 0:  # ensure there are pixels
                neuropil_trace = neuropil_masked_data.mean(axis=1)
                # Apply correction to roi_trace for downstream analysis
                roi_trace = roi_trace - neuropil_correction_factor * neuropil_trace
            else:
                cali_logger.warning(
                    f"No neuropil pixels found for ROI {label_value} in {fov_name}"
                )

        # calculate the dff of the roi trace
        # (using corrected trace if neuropil is enabled)
        # Try to get frame rate from settings first, then from metadata
        if extraction_settings.frame_rate is not None:
            frame_rate = extraction_settings.frame_rate
        else:
            exposure_ms = meta[0].get("exposure_ms", None)
            if exposure_ms is None:
                msg = (
                    "❌ Frame rate or exposure time must be provided in "
                    "extraction settings or metadata."
                )
                cali_logger.error(msg)
                raise ValueError(msg)
            frame_rate = 1000.0 / exposure_ms

        # Acquire numba lock for thread-safe numba computation
        with _NUMBA_LOCK:
            dff = calculate_dff(
                roi_trace,
                window_sec=extraction_settings.dff_window,
                frame_rate=frame_rate,
                percentile=extraction_settings.dff_percentile,
            )

        # Check for cancellation after DFF calculation
        if self._check_for_abort_requested():
            return None

        # run OASIS deconvolution on the dff trace
        tau = extraction_settings.decay_constant or 0.0  # seconds
        frame_rate = len(dff) / tot_time_sec  # Hz
        if tau > 0.0:
            # User-provided decay constant τ → AR(1) coefficient g
            g1 = float(np.exp(-1.0 / (frame_rate * tau)))  # AR(1) coefficient
            g: tuple[float] | tuple[float, float] = (g1,)  # OASIS expects a tuple
            optimize_g = 0  # do NOT re-optimize g, user fixed it
            # Estimate noise from DFF trace
            sn = GetSn(dff, range_ff=[0.25, 0.5], method="median")
        else:
            # Estimate AR parameters + noise from ORIGINAL dff trace
            try:
                g_arr, sn = estimate_parameters(
                    dff,
                    p=1,  # AR(1)
                    range_ff=[0.25, 0.5],  # default
                    method="median",  # mean or logmexp
                    lags=10,
                    fudge_factor=0.98,
                )
                # make sure g is a tuple
                g = tuple(np.atleast_1d(g_arr))
            except (np.linalg.LinAlgError, ValueError) as e:
                # estimate_parameters can fail with LinAlgError on edge-case traces
                # (e.g., very short traces or numerical issues in autocorrelation)
                cali_logger.warning(
                    f"⚠️ OASIS parameter estimation failed for ROI {label_value} in "
                    f"{fov_name}: {e}. Using default AR1 coefficient (0.95,)."
                )
                g = (0.95,)  # fallback stable AR(1) coefficient
                sn = GetSn(dff, range_ff=[0.25, 0.5], method="median")
            optimize_g = 0  # set >0 only if you really want refine

        # Deconvolve with error handling for invalid AR coefficients
        try:
            den_dff, spikes, _b, _g_fit, _lam = deconvolve(
                dff,
                g=g,
                sn=sn,
                penalty=1,  # L1 sparsity (standard OASIS)
                optimize_g=optimize_g,
            )
        except (ValueError, RuntimeError) as e:
            # If OASIS fails due to invalid AR coefficients, fall back to stable default
            cali_logger.warning(
                f"⚠️ OASIS deconvolution failed for ROI {label_value} in {fov_name}: "
                f"{e}. Retrying with stable default AR1 coefficient (0.95,)."
            )
            optimize_g = 0
            den_dff, spikes, _b, _g_fit, _lam = deconvolve(
                dff,
                g=(0.95,),  # fallback stable AR(1) coefficient
                sn=sn,
                penalty=1,  # L1 sparsity (standard OASIS)
                optimize_g=optimize_g,
            )

        cali_logger.debug(f"OASIS params ROI {label_value}: g={g}, sn={sn}")

        # Convert to float
        den_dff = den_dff.astype(float)
        spikes = spikes.astype(float)

        # Check for cancellation after deconvolution
        if self._check_for_abort_requested():
            return None

        # Create Traces object (extraction product)
        corrected_trace = (
            cast("list[float]", roi_trace.tolist())
            if neuropil_trace is not None
            else None
        )
        traces = Traces(
            raw_trace=cast("list[float]", roi_trace_uncorrected.tolist()),
            corrected_trace=corrected_trace,
            neuropil_trace=(
                cast("list[float]", neuropil_trace.tolist())
                if neuropil_trace is not None
                else None
            ),
            dff=cast("list[float]", dff.tolist()),
            den_dff=den_dff.tolist(),
            inferred_spikes=spikes.tolist(),
            x_axis=elapsed_time_list,
            x_axis_units=x_unit,
        )

        # Optionally perform analysis (peak detection, IEI, frequency)
        data_analysis = None
        active = False
        stimulated = False

        if analysis_settings is not None:
            # Import analysis functions
            from cali.analysis._trace_analysis import (
                calculate_frequency,
                calculate_inter_event_intervals,
                compute_calcium_peak_detection_thresholds,
                count_thresholded_spike_events,
                detect_peaks_in_trace,
            )

            # check if the roi is stimulated
            roi_stimulation_overlap_ratio = 0.0
            stimulated_area_mask = analysis_settings.stimulated_mask_area()
            if stimulated_area_mask is not None:
                roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
                    stimulated_area_mask, label_mask
                )
            # consider the roi stimulated if more than 10% of the roi overlaps
            stimulated = roi_stimulation_overlap_ratio > 0.1

            # --- Calcium peak detection (gated by enable_calcium) ---
            frequency = None
            peaks_den_dff = np.array([], dtype=int)
            peaks_amplitudes_den_dff: list[float] = []
            iei: list[float] = []
            peaks_prominence_den_dff = None
            peaks_height_den_dff = None

            if analysis_settings.enable_calcium:
                # fmt: off
                peaks_height_den_dff, peaks_prominence_den_dff = compute_calcium_peak_detection_thresholds(den_dff, sn, analysis_settings)  # noqa E501
                # fmt: on

                if self._check_for_abort_requested():
                    return None

                min_distance_ms = analysis_settings.peaks_distance
                min_distance_frames = max(
                    1, int((min_distance_ms / 1000.0) * analysis_settings.frame_rate)
                )
                peaks_den_dff, peaks_amplitudes_den_dff = detect_peaks_in_trace(
                    den_dff,
                    peaks_height_den_dff,
                    peaks_prominence_den_dff,
                    min_distance_frames,
                )

                if self._check_for_abort_requested():
                    return None

                frequency = calculate_frequency(len(peaks_den_dff), tot_time_sec)
                iei_ms = calculate_inter_event_intervals(
                    peaks_den_dff, elapsed_time_list
                )
                iei = [x / 1000 for x in iei_ms]

            # --- Inferred spike analysis (gated by enable_spikes) ---
            spike_detection_threshold = None
            inferred_spikes_freq = None
            inferred_spikes_rising_edge_freq = None
            num_thresholded_spikes = 0

            if analysis_settings.enable_spikes:
                # fmt: off
                spike_detection_threshold = compute_inferred_spike_threshold(spikes, analysis_settings)  # noqa E501
                # fmt: on
                num_thresholded_spikes, num_rising_edges = (
                    count_thresholded_spike_events(spikes, spike_detection_threshold)
                )
                inferred_spikes_freq = calculate_frequency(
                    num_thresholded_spikes, tot_time_sec
                )
                if analysis_settings.enable_rising_edge_analysis:
                    inferred_spikes_rising_edge_freq = calculate_frequency(
                        num_rising_edges, tot_time_sec
                    )

            # Create DataAnalysis object (analysis product)
            data_analysis = DataAnalysis(
                total_recording_time_sec=tot_time_sec,
                den_dff_frequency=frequency,
                peaks_den_dff=(
                    peaks_den_dff.tolist() if len(peaks_den_dff) > 0 else None
                ),
                peaks_amplitudes_den_dff=peaks_amplitudes_den_dff or None,
                iei=iei or None,
                peaks_prominence_den_dff=peaks_prominence_den_dff,
                peaks_height_den_dff=peaks_height_den_dff,
                inferred_spikes_threshold=spike_detection_threshold,
                inferred_spikes_frequency=inferred_spikes_freq,
                inferred_spikes_rising_edge_frequency=inferred_spikes_rising_edge_freq,
            )

            # Determine active status based on enabled analyses
            if analysis_settings.enable_calcium:
                active = len(peaks_den_dff) > 0
            elif analysis_settings.enable_spikes:
                active = num_thresholded_spikes > 0
            else:
                active = False

        return (traces, data_analysis, active, stimulated, roi_size, roi_size_units)
