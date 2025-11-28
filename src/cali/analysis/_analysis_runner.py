"""Analysis runner for computing metrics from extracted traces.

This module provides the AnalysisRunner class that takes FOVs with ROIs
containing Traces data and computes analysis metrics (peaks, IEI, frequency).
"""

import threading
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Callable

import numpy as np
from tqdm import tqdm

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, AnalysisSettings, DataAnalysis

if TYPE_CHECKING:
    from cali.sqlmodel._model import ROI, Traces
from ._trace_analysis import (
    calculate_frequency,
    calculate_inter_event_intervals,
    compute_peak_detection_thresholds,
    detect_peaks_in_trace,
)
from ._util import get_overlap_roi_with_stimulated_area


class AnalysisRunner:
    """Runner for analyzing extracted trace data.

    Takes FOVs with ROIs containing Traces and computes DataAnalysis metrics:
    - Peak detection in deconvolved traces
    - Inter-event intervals
    - Event frequencies
    - Peak amplitudes

    This allows re-running analysis with different parameters without
    re-extracting traces from imaging data.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cancellation_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the analysis process."""
        cali_logger.info("🗑️ Analysis cancellation requested...")
        self._cancellation_event.set()

    def run(
        self,
        fovs: Iterable[FOV],
        analysis_settings: AnalysisSettings,
        as_generator: bool = False,
    ) -> Generator[FOV, None, None] | list[FOV]:
        """Run analysis on FOVs with existing Traces data.

        Parameters
        ----------
        fovs : Iterable[FOV]
            FOVs with ROIs containing Traces to analyze
        analysis_settings : AnalysisSettings
            Analysis parameters (peak detection thresholds, etc.)
        as_generator : bool
            If True, returns a Generator that yields FOVs.
            If False (default), returns a list of FOVs.

        Returns
        -------
        Generator[FOV, None, None] | list[FOV]
            FOV objects with ROIs containing DataAnalysis results,
            ready to be saved to database
        """
        generator = self._run_generator(fovs, analysis_settings)
        return generator if as_generator else list(generator)

    def _run_generator(
        self,
        fovs: Iterable[FOV],
        analysis_settings: AnalysisSettings,
    ) -> Generator[FOV, None, None]:
        """Internal generator for analysis process."""
        self._cancellation_event.clear()

        cali_logger.info(f"⚡️ Using {analysis_settings.threads} threads")

        # Execute analysis in parallel and yield results
        for fov_result in self._exec_in_threadpool(
            analyze=self._analyze_fov,
            cancel_event=self._cancellation_event,
            fovs=fovs,
            analysis_settings=analysis_settings,
            max_workers=analysis_settings.threads,
        ):
            if fov_result is not None:
                yield fov_result

        cali_logger.info("✅ Analysis complete!")

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_event.is_set()

    def _exec_in_threadpool(
        self,
        analyze: Callable,
        cancel_event: threading.Event,
        fovs: Iterable[FOV],
        analysis_settings: AnalysisSettings,
        max_workers: int | None = None,
    ) -> Iterable[FOV]:
        """Execute analysis in parallel and yield FOV results."""
        # If max_workers is 1, run in the main thread to avoid overhead and potential
        # threading issues
        if max_workers == 1:
            for fov in fovs:
                if cancel_event.is_set():
                    cali_logger.info("🚮 Cancellation requested")
                    break
                try:
                    if (
                        fov_result := analyze(analysis_settings, fov)
                    ) is not None:
                        yield fov_result
                except Exception:
                    import traceback

                    full_tb = traceback.format_exc()
                    cali_logger.error(f"Exception in analysis: {full_tb}")

            if cancel_event.is_set():
                cali_logger.info("❌ Run Cancelled")

            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if cancel_event.is_set():
                cali_logger.info("🚮 Cancellation requested before starting analysis")
                return

            futures = (
                executor.submit(
                    analyze,
                    analysis_settings,
                    fov,
                )
                for fov in fovs
            )

            for future in as_completed(futures):
                # Check for cancellation at the start of each iteration
                if cancel_event.is_set():
                    cali_logger.info(
                        "🚮 Cancellation requested, shutting down executor..."
                    )
                    # Cancel pending futures and shutdown executor
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                try:
                    # Commit the results to database if we got any
                    if (fov_result := future.result()) is not None:
                        yield fov_result
                except Exception:
                    import traceback

                    full_tb = traceback.format_exc()
                    cali_logger.error(f"Exception in analysis thread: {full_tb}")

        if cancel_event.is_set():
            cali_logger.info("❌ Analysis Cancelled")

    def _analyze_fov(
        self,
        analysis_settings: AnalysisSettings,
        fov: FOV,
    ) -> FOV | None:
        """Analyze all ROIs in a FOV.

        Parameters
        ----------
        analysis_settings : AnalysisSettings
            Analysis parameters
        fov : FOV
            FOV with ROIs containing Traces

        Returns
        -------
        FOV | None
            FOV with DataAnalysis added to ROIs, or None if cancelled
        """
        if self._check_for_abort_requested():
            return None

        msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - "
            f"cali_logger - INFO - 📊 Analyzing Traces for {fov.name}."
        )

        # cali_logger.info(msg)

        for roi in tqdm(fov.rois, desc=msg):
            if self._check_for_abort_requested():
                cali_logger.info(
                    f"🚮 Cancellation requested during analysis of {fov.name}"
                )
                break

            # Skip ROIs without traces
            if not roi.traces_history:
                cali_logger.warning(
                    f"ROI {roi.label_value} in {fov.name} has no traces. "
                    "Run extraction first."
                )
                continue

            # Get the most recent traces (last in list)
            traces = roi.traces_history[-1]

            # Analyze the traces
            analysis_data = self._analyze_roi_traces(
                traces=traces, analysis_settings=analysis_settings, roi=roi
            )

            if analysis_data is not None:
                data_analysis, active, stimulated = analysis_data

                # Store analysis in temporary list (similar to extraction pattern)
                if not hasattr(roi, "_new_data_analysis"):
                    roi._new_data_analysis = []  # type: ignore
                roi._new_data_analysis.append(data_analysis)  # type: ignore
                roi.active = active
                roi.stimulated = stimulated

        return fov

    def _analyze_roi_traces(
        self,
        traces: "Traces",
        analysis_settings: AnalysisSettings,
        roi: "ROI",
    ) -> tuple[DataAnalysis, bool, bool] | None:
        """Analyze traces for a single ROI.

        Parameters
        ----------
        traces : Traces
            Traces object with dec_dff, inferred_spikes, etc.
        analysis_settings : AnalysisSettings
            Analysis parameters
        roi : ROI
            ROI object containing the mask for stimulation overlap check

        Returns
        -------
        tuple[DataAnalysis, bool, bool] | None
            (DataAnalysis, active, stimulated) or None if processing fails.
        """
        if self._check_for_abort_requested():
            return None

        # Convert traces to numpy arrays
        dec_dff = np.array(traces.dec_dff)
        spikes = np.array(traces.inferred_spikes)
        elapsed_time_list = traces.x_axis

        # Skip if no time axis data
        if elapsed_time_list is None or len(elapsed_time_list) < 2:
            cali_logger.warning("Traces missing time axis data, skipping analysis")
            return None

        # Calculate total recording time
        tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0]) / 1000

        # Compute thresholds
        (
            peaks_height_dec_dff,
            peaks_prominence_dec_dff,
            spike_detection_threshold,
        ) = compute_peak_detection_thresholds(dec_dff, spikes, analysis_settings)

        if self._check_for_abort_requested():
            return None

        # Detect peaks
        min_distance_frames = analysis_settings.peaks_distance
        peaks_dec_dff, peaks_amplitudes_dec_dff = detect_peaks_in_trace(
            dec_dff,
            peaks_height_dec_dff,
            peaks_prominence_dec_dff,
            min_distance_frames,
        )

        if self._check_for_abort_requested():
            return None

        # Calculate frequency
        frequency = calculate_frequency(len(peaks_dec_dff), tot_time_sec)

        # Calculate IEI (convert from ms to sec)
        iei_ms = calculate_inter_event_intervals(peaks_dec_dff, elapsed_time_list)
        iei = [x / 1000 for x in iei_ms]  # Convert ms to sec

        # Check if the ROI is stimulated (evoked experiments only)
        stimulated = False
        stimulated_area_mask = analysis_settings.stimulated_mask_area()
        if stimulated_area_mask is not None and roi.roi_mask is not None:
            # Get label mask from ROI mask coordinates
            from cali.util import coordinates_to_mask

            if (
                roi.roi_mask.coords_y is not None
                and roi.roi_mask.coords_x is not None
                and roi.roi_mask.height is not None
                and roi.roi_mask.width is not None
            ):
                label_mask = coordinates_to_mask(
                    (roi.roi_mask.coords_y, roi.roi_mask.coords_x),
                    (roi.roi_mask.height, roi.roi_mask.width),
                )
                roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
                    stimulated_area_mask, label_mask
                )
                # Consider the ROI stimulated if more than 10% overlaps
                stimulated = roi_stimulation_overlap_ratio > 0.1

        # Create DataAnalysis object
        data_analysis = DataAnalysis(
            total_recording_time_sec=tot_time_sec,
            dec_dff_frequency=frequency,
            peaks_dec_dff=peaks_dec_dff.tolist(),
            peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
            iei=iei,
            peaks_prominence_dec_dff=peaks_prominence_dec_dff,
            peaks_height_dec_dff=peaks_height_dec_dff,
            inferred_spikes_threshold=spike_detection_threshold,
        )

        active = len(peaks_dec_dff) > 0

        return (data_analysis, active, stimulated)
