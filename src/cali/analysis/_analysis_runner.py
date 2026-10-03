"""Analysis runner for computing metrics from extracted traces.

This module provides the AnalysisRunner class that takes FOVs with ROIs
containing Traces data and computes analysis metrics (peaks, IEI, frequency).
"""

import threading
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from tqdm import tqdm

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, AnalysisSettings, DataAnalysis
from cali.sqlmodel._spike_settings import require_available_spike_methods

if TYPE_CHECKING:
    from cali.sqlmodel._model import ROI, Traces
from ._fov_metrics import get_overlap_roi_with_stimulated_area
from ._roi_analysis import AnalysisCancelled, analyze_roi_traces


class AnalysisRunner:
    """Runner for analyzing extracted trace data.

    Takes FOVs with ROIs containing Traces and computes DataAnalysis metrics:
    - Peak detection in denoised traces
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
        cali_logger.info("🚮 Analysis cancellation requested...")
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
        analysis_settings.validate_spike_settings()
        if analysis_settings.enable_spikes:
            require_available_spike_methods(
                tuple(child.method for child in analysis_settings.spike_settings)
            )
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

        # Phase 1: Execute ROI analysis in parallel threads
        # Collect FOVs that need FOV-level analysis
        fovs_for_analysis: list[FOV] = []

        for fov_result in self._exec_in_threadpool(
            analyze=self._analyze_fov,
            cancel_event=self._cancellation_event,
            fovs=fovs,
            analysis_settings=analysis_settings,
            max_workers=analysis_settings.threads,
        ):
            if fov_result is not None:
                # Check if FOV needs FOV-level analysis
                if hasattr(fov_result, "_pending_analysis_settings"):
                    fovs_for_analysis.append(fov_result)
                else:
                    yield fov_result

        # Phase 2: Compute FOV-level analysis SEQUENTIALLY
        # This uses multiprocessing internally for CCG pairs, which is much faster
        # than having multiple threads each create their own Pool
        if fovs_for_analysis and not self._cancellation_event.is_set():
            from cali.analysis._fov_analysis_parallel import (
                compute_fov_analysis_parallel,
            )

            cali_logger.info(
                f"📊 Computing FOV-level analysis for {len(fovs_for_analysis)} FOVs..."
            )
            for fov_result in fovs_for_analysis:
                if self._cancellation_event.is_set():
                    break

                pending_settings = fov_result._pending_analysis_settings
                delattr(fov_result, "_pending_analysis_settings")

                cali_logger.info(f"📊 Analyzing {fov_result.name}...")
                fov_analysis = compute_fov_analysis_parallel(
                    fov_result, pending_settings
                )
                if fov_analysis is not None:
                    if not hasattr(fov_result, "_new_fov_analysis"):
                        fov_result._new_fov_analysis = []
                    fov_result._new_fov_analysis.append(fov_analysis)
                cali_logger.info(
                    f"✅ FOV-level analysis complete for {fov_result.name}."
                )
                yield fov_result

        if self._cancellation_event.is_set():
            cali_logger.info("🛑 Analysis Cancelled!")
        else:
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
            traces = getattr(roi, "_analysis_source_trace", None)
            if traces is None:
                traces = roi.traces_history[-1]

            # Analyze the traces
            analysis_data = self._analyze_roi_traces(
                traces=traces, analysis_settings=analysis_settings, roi=roi
            )

            if analysis_data is not None:
                data_analysis, active, stimulated = analysis_data

                # Store analysis in temporary list (similar to extraction pattern)
                if not hasattr(roi, "_new_data_analysis"):
                    roi._new_data_analysis = []
                roi._new_data_analysis.append(data_analysis)
                roi.active = active
                roi.stimulated = stimulated

        # NOTE: FOV-level analysis (CCG) is now computed AFTER the threadpool completes
        # in _run_generator(). This avoids concurrent Pool creation when multiple
        # threads call compute_fov_analysis_parallel simultaneously.
        # Store analysis_settings on FOV for later use.
        fov._pending_analysis_settings = analysis_settings

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
            Traces object with den_dff, inferred_spikes, etc.
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

        elapsed_time_list = traces.x_axis
        if elapsed_time_list is None or len(elapsed_time_list) < 2:
            cali_logger.warning("Traces missing time axis data, skipping analysis")
            return None
        tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0]) / 1000
        try:
            data_analysis = analyze_roi_traces(
                traces,
                analysis_settings,
                duration_s=tot_time_sec,
                cancel=self._check_for_abort_requested,
            )
        except AnalysisCancelled:
            return None

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

        active = bool(data_analysis.calcium_active) or any(
            child.spike_active for child in data_analysis.spike_analyses
        )

        return (data_analysis, active, stimulated)
