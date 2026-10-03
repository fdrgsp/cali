import threading

# ignore deprecation warnings from oasis
import warnings
from collections import deque
from collections.abc import Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from typing import Callable, cast

import numpy as np
from tqdm import tqdm

from cali._constants import EVENT_KEY
from cali.analysis._fov_metrics import get_overlap_roi_with_stimulated_area
from cali.analysis._trace_analysis import compute_inferred_spike_threshold
from cali.logger import cali_logger
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.readers._tiff_collection_reader import TiffCollectionReader
from cali.sqlmodel._model import (
    FOV,
    AnalysisSettings,
    DataAnalysis,
    ExtractionFrameWindow,
    ExtractionSettings,
    Mask,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
)
from cali.sqlmodel._spike_settings import require_available_spike_methods
from cali.util import coordinates_to_mask, mask_to_coordinates
from cali.util._util import _NUMBA_LOCK

from ._frame_window import (
    ExtractionFrameWindow as ResolvedFrameWindow,
)
from ._frame_window import (
    StartupDiscardError,
    build_timing_descriptor,
    resolve_initial_frame_window,
    retained_time_axis,
)
from ._neuropil import create_neuropil_from_dilation
from ._spike_inference import InferenceCancelled, OasisBackend
from ._util import calculate_dff

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class _RoiParts:
    """Minimal ROI data retained between DFF computation and finalization."""

    label_value: int
    label_mask: np.ndarray
    raw: np.ndarray
    corrected: np.ndarray | None
    neuropil: np.ndarray | None
    dff: np.ndarray
    roi_size: float
    roi_size_units: str


class ExtractionRunner:
    def __init__(self) -> None:
        super().__init__()

        # Use threading.Event for cancellation control
        self._cancellation_event = threading.Event()

    # -------------------------PUBLIC METHODS-----------------------------------

    def cancel(self) -> None:
        """Request cancellation of the extraction process."""
        cali_logger.info("🚮 Cancellation requested...")
        self._cancellation_event.set()

    def run(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        extraction_settings: ExtractionSettings,
        fovs: Iterable[FOV],
        *,
        analysis_settings: AnalysisSettings | None = None,
        as_generator: bool = False,
    ) -> Generator[FOV, None, None] | list[FOV]:
        """Run extraction and optionally analysis on FOVs with ROIs.

        This method performs pure computation - it takes FOVs with ROIs and adds
        traces and optionally analysis data to them.
        It does not interact with the database.

        Parameters
        ----------
        dataset : TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader
            Data reader instance for imaging data
        extraction_settings : ExtractionSettings
            Extraction parameters (neuropil, dff_window, decay_constant, etc.)
        fovs : Iterable[FOV]
            FOVs with ROIs to analyze. These typically come from DetectionRunner
            or are loaded from the database by the caller.
        analysis_settings : AnalysisSettings | None
            Analysis parameters (peak detection, thresholds, etc.)
            If set, perform peak analysis after extraction.
        as_generator : bool
            If True, returns a Generator that yields FOVs.
            If False (default), returns a list of FOVs.

        Returns
        -------
        Generator[FOV, None, None] | list[FOV]
            FOV objects with ROIs containing traces and optionally analysis data,
            ready to be saved to database

        Raises
        ------
        ValueError
            If no ROI masks are found in the provided FOVs
        ValueError
            If run_analysis=True but analysis_settings is None
        """
        extraction_settings.validate_output_settings()
        require_available_spike_methods(extraction_settings.spike_methods)
        if analysis_settings is not None:
            analysis_settings.validate_spike_settings(extraction_settings.spike_methods)
        generator = self._run_generator(
            dataset, extraction_settings, analysis_settings, fovs
        )

        return generator if as_generator else list(generator)

    def _run_generator(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        extraction_settings: ExtractionSettings,
        analysis_settings: AnalysisSettings | None,
        fovs: Iterable[FOV],
    ) -> Generator[FOV, None, None]:
        """Internal generator for analysis process."""
        # Reset cancellation event
        self._cancellation_event.clear()

        assert isinstance(
            dataset, (TensorstoreZarrReader, OMEZarrReader, TiffCollectionReader)
        ), (
            "Data must be a TensorstoreZarrReader, OMEZarrReader, or "
            "TiffCollectionReader instance."
        )

        # Eagerly load stimulation_mask attributes to prevent lazy loading issues
        # in thread pool workers
        if analysis_settings is not None and analysis_settings.stimulation_mask:
            # Access the attributes to force SQLAlchemy to load them
            _ = analysis_settings.stimulation_mask.coords_y
            _ = analysis_settings.stimulation_mask.coords_x
            _ = analysis_settings.stimulation_mask.height
            _ = analysis_settings.stimulation_mask.width

        cali_logger.info(f"⚡️ Using {extraction_settings.threads} threads")

        # Phase 1: Execute extraction in parallel threads
        # Collect FOVs that need FOV-level analysis
        fovs_for_analysis: list[FOV] = []

        for fov_result in self._exec_in_threadpool(
            analyze=self._analyze_position,
            dataset=dataset,
            cancel_event=self._cancellation_event,
            fovs=fovs,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings,
            max_workers=extraction_settings.threads,
        ):
            if fov_result is not None:
                # Check if FOV needs analysis
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

        if analysis_settings is None:
            if self._cancellation_event.is_set():
                msg = "🛑 Extraction Cancelled!"
            else:
                msg = "✅ Extraction complete!"
        else:
            if self._cancellation_event.is_set():
                msg = "🛑 Extraction and Analysis Cancelled!"
            else:
                msg = "✅ Extraction and Analysis complete!"
        cali_logger.info(msg)

    # -------------------------PRIVATE METHODS-----------------------------------

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_event.is_set()

    def _exec_in_threadpool(
        self,
        analyze: Callable,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        cancel_event: threading.Event,
        fovs: Iterable[FOV],
        extraction_settings: ExtractionSettings,
        analysis_settings: AnalysisSettings | None,
        max_workers: int | None = None,
    ) -> Iterable[FOV]:
        """Execute extraction in parallel and yield FOV results."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Check for cancellation before submitting futures
            if cancel_event.is_set():
                cali_logger.info(
                    "🚮 Cancellation requested before starting thread pool"
                )
                return

            # Convert generator to list to get all futures at once
            futures = [
                executor.submit(
                    analyze,
                    dataset,
                    extraction_settings,
                    analysis_settings,
                    fov,
                )
                for fov in fovs
            ]

            # Process completed futures with cancellation checks
            # Use a timeout to periodically check for cancellation even if no futures
            # have completed
            completed_futures: set[Future] = set()
            while len(completed_futures) < len(futures):
                # Check for cancellation before waiting for futures
                if cancel_event.is_set():
                    cali_logger.info(
                        "🚮 Cancellation requested, shutting down executor..."
                    )
                    # Cancel pending futures and shutdown executor
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                # Wait for futures with short timeout to enable responsive cancellation
                done, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)

                # Process newly completed futures
                for future in done:
                    if future in completed_futures:
                        continue
                    completed_futures.add(future)

                    try:
                        # Commit the results to database if we got any
                        if (fov_result := future.result()) is not None:
                            yield fov_result
                    except StartupDiscardError:
                        self._cancellation_event.set()
                        for pending in futures:
                            pending.cancel()
                        raise
                    except Exception:
                        import traceback

                        full_tb = traceback.format_exc()
                        cali_logger.error(f"Exception in extraction thread: {full_tb}")

    def _analyze_position(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        extraction_settings: ExtractionSettings,
        analysis_settings: AnalysisSettings | None,
        fov: FOV,
    ) -> FOV | None:
        """Extract the roi traces for the given position and return result objects.

        Returns a FOV object with all its nested relationships
        (ROIs, Traces, DataAnalysis, Masks) ready to be committed.
        """
        return self._extract_trace_data_per_position(
            dataset,
            extraction_settings,
            analysis_settings,
            fov,
        )

    def _extract_trace_data_per_position(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        extraction_settings: ExtractionSettings,
        analysis_settings: AnalysisSettings | None,
        fov_to_analyze: FOV,
    ) -> FOV | None:
        """Extract trace data for a position and return FOV objects (not committed).

        Returns a FOV with all its ROIs, Traces, DataAnalysis, and Masks
        ready to be committed to the database.
        """
        # if runner._data is None or runner._check_for_abort_requested():
        if self._check_for_abort_requested():
            return None

        extraction_settings.validate_output_settings()
        require_available_spike_methods(extraction_settings.spike_methods)
        if analysis_settings is not None:
            analysis_settings.validate_spike_settings(extraction_settings.spike_methods)

        global_pos_idx = fov_to_analyze.position_index

        # get the data and metadata for the position
        data, meta = dataset.isel(p=global_pos_idx, metadata=True)
        # get the fov_name name from metadata
        fov_name = self._get_fov_name(EVENT_KEY, meta, global_pos_idx)

        original_frame_count = data.shape[0]
        timing = build_timing_descriptor(meta, original_frame_count)
        frame_window = resolve_initial_frame_window(
            discard_value=extraction_settings.discard_initial_value,
            discard_unit=extraction_settings.discard_initial_unit,
            frame_rate=extraction_settings.frame_rate,
            frame_rate_verified=extraction_settings.frame_rate_verified,
            timing=timing,
        )
        if frame_window.source_start_frame:
            cali_logger.info(
                f"⏭️ {fov_name}: discarded {frame_window.source_start_frame} of "
                f"{frame_window.original_frame_count} startup frames "
                f"({frame_window.discarded_duration_ms / 1000.0:.3f} s; "
                f"{frame_window.timing_source} timing)"
            )
            data = data[frame_window.source_start_frame :]
            # Some readers return one metadata item per frame; others return a single
            # static metadata record. Only crop genuinely per-frame metadata.
            if len(meta) == original_frame_count:
                meta = meta[frame_window.source_start_frame :]

        elapsed_time_list = retained_time_axis(
            timing, frame_window, frame_rate=extraction_settings.frame_rate
        )

        if fov_to_analyze is None or not fov_to_analyze.rois:
            cali_logger.error(
                f"No ROI masks found for FOV {fov_name} at position {global_pos_idx}. "
                "Run detection first."
            )
            return None

        # Convert ROI masks to numpy arrays: {label_value: np.ndarray mask}
        labels_masks = self._get_label_mask(fov_to_analyze, fov_name)

        if not labels_masks:
            cali_logger.error(
                f"No valid ROI masks found for FOV {fov_name}. Run detection first."
            )
            return None

        # Check for cancellation after loading and processing labels
        if self._check_for_abort_requested():
            return None

        # Prepare masks for neuropil correction if enabled
        labels_masks, neuropil_masks_dict = self._prepare_neuropil_masks(
            extraction_settings, data, labels_masks
        )

        # get the total time in seconds for the recording
        tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0]) / 1000

        # Use the existing FOV from detection (don't create a new one)
        # We'll add traces to the existing ROIs

        msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - "
            f"cali_logger - INFO - 📈 Extracting Traces Data from {fov_name}."
        )

        # Create a map of label_value -> ROI for quick lookup
        roi_map = {roi.label_value: roi for roi in fov_to_analyze.rois}

        parts_by_roi: deque[_RoiParts] = deque()
        for label_value in tqdm(labels_masks.keys(), desc=msg):
            if self._check_for_abort_requested():
                cali_logger.info(
                    f"🚮 Cancellation requested during processing of {fov_name}"
                )
                return None

            # Get the existing ROI from detection
            existing_roi = roi_map.get(label_value)
            if existing_roi is None:
                cali_logger.warning(
                    f"No ROI found with label_value={label_value} in {fov_name}"
                )
                continue

            # Process the trace and get Traces + DataAnalysis objects
            neuropil_correction_factor = (
                extraction_settings.neuropil_correction_factor
                if (
                    extraction_settings.neuropil_inner_radius > 0
                    and extraction_settings.neuropil_min_pixels > 0
                )
                else None
            )
            parts = self._compute_roi_dff(
                data,
                meta,
                fov_name,
                extraction_settings,
                label_value,
                labels_masks[label_value],
                neuropil_masks_dict.get(label_value),
                neuropil_correction_factor,
            )

            if parts is not None:
                parts_by_roi.append(parts)

        # Phase B: exactly one backend invocation per FOV. The legacy rate is
        # deliberately retained here to preserve OASIS AR(1) numerics.
        if self._check_for_abort_requested():
            return None
        if parts_by_roi:
            dff_matrix = np.vstack([parts.dff for parts in parts_by_roi])
            try:
                oasis = OasisBackend().infer_all(
                    dff_matrix,
                    frame_rate=dff_matrix.shape[1] / tot_time_sec,
                    decay_constant=extraction_settings.decay_constant,
                    roi_labels=[parts.label_value for parts in parts_by_roi],
                    fov_name=fov_name,
                    cancel=self._check_for_abort_requested,
                )
            except InferenceCancelled:
                return None
            del dff_matrix

        # One persisted transform per FOV; all base traces reference this row.
        origin = float(timing.timestamps_ms[0]) if timing.timestamps_ms else None
        stored_window = ExtractionFrameWindow(
            fov_id=fov_to_analyze.id,
            requested_discard_value=extraction_settings.discard_initial_value,
            requested_discard_unit=extraction_settings.discard_initial_unit,
            timing_source=frame_window.timing_source,
            conversion_rule=(
                "frames"
                if extraction_settings.discard_initial_unit == "frames"
                else "timestamps"
                if timing.trusted
                else "verified_frame_rate"
            ),
            original_frame_count=frame_window.original_frame_count,
            retained_frame_count=frame_window.retained_frame_count,
            source_start_frame=frame_window.source_start_frame,
            source_start_time_ms=frame_window.source_start_time_ms,
            source_time_origin_ms=origin,
            source_start_timestamp_ms=(
                origin + frame_window.source_start_time_ms
                if origin is not None
                else None
            ),
            discarded_duration_ms=frame_window.discarded_duration_ms,
            provenance_source="extraction",
        )
        inference_run = SpikeInferenceRun(
            backend_version=version("oasis-deconv"),
            resolved_device="cpu",
            dtype="float64",
            provenance_source="oasis_inference",
        )

        # Phase C: stage complete products before attaching anything to the FOV.
        finalized = []
        for index in range(len(parts_by_roi)):
            if self._check_for_abort_requested():
                return None
            parts = parts_by_roi.popleft()
            trace_data = self._finalize_roi(
                parts,
                oasis.den_dff[index],
                oasis.spikes[index],
                float(oasis.sn_by_roi[index]),
                analysis_settings,
                tot_time_sec,
                elapsed_time_list,
                "ms",
                frame_window,
                stored_window=stored_window,
                inference_run=inference_run,
                ar_coefficients=oasis.g_by_roi[index].tolist(),
            )
            if trace_data is None:
                return None
            finalized.append((parts.label_value, trace_data))
            # Release Phase-A traces immediately after conversion to stored lists.
            del parts
        if self._check_for_abort_requested():
            return None

        for label_value, trace_data in finalized:
            existing_roi = roi_map[label_value]
            (
                traces,
                data_analysis,
                active,
                stimulated,
                roi_size,
                roi_size_units,
            ) = trace_data
            # Store cell size in ROI (update on every extraction run)
            # This ensures the value is always populated even if initially None
            existing_roi.cell_size = roi_size
            existing_roi.cell_size_units = roi_size_units

            # Save neuropil mask to the Traces object if it exists for this ROI
            neuropil_mask_array = neuropil_masks_dict.get(label_value)
            if neuropil_mask_array is not None and neuropil_mask_array.any():
                # Convert mask to sparse coordinates
                neuropil_coords, neuropil_shape = mask_to_coordinates(
                    neuropil_mask_array
                )
                # Create Mask object
                neuropil_mask_obj = Mask(
                    coords_y=neuropil_coords[0],
                    coords_x=neuropil_coords[1],
                    height=neuropil_shape[0],
                    width=neuropil_shape[1],
                    mask_type="neuropil",
                )
                # Assign to Traces (will be saved via relationship cascade)
                traces.neuropil_mask = neuropil_mask_obj

            # Store new traces/analysis in temporary list on ROI
            # This avoids SQLAlchemy warnings about modifying collections
            # during threaded execution. The commit function will handle
            # proper attachment.
            if not hasattr(existing_roi, "_new_traces"):
                existing_roi._new_traces = []
                existing_roi._new_data_analysis = []
            existing_roi._new_traces.append(traces)
            # Only add data_analysis if it was computed
            if data_analysis is not None:
                existing_roi._new_data_analysis.append(data_analysis)
            existing_roi.active = active
            existing_roi.stimulated = stimulated

        # NOTE: FOV-level analysis (CCG) is now computed AFTER the threadpool completes
        # in _run_generator(). This avoids concurrent Pool creation when multiple
        # threads call compute_fov_analysis_parallel simultaneously.
        # The analysis_settings is stored on the FOV for later use.
        if analysis_settings is not None:
            fov_to_analyze._pending_analysis_settings = analysis_settings

        # Return the FOV with updated ROIs (will be committed by caller)
        return fov_to_analyze

    def _get_fov_to_analyze(
        self,
        global_pos_idx: int,
        fovs_with_rois: list[FOV],
    ) -> FOV | None:
        """Get the FOV to analyze for the given position index."""
        for fov in fovs_with_rois:
            if fov.position_index == global_pos_idx:
                return fov
        return None

    def _prepare_neuropil_masks(
        self,
        extraction_settings: ExtractionSettings,
        data: np.ndarray,
        labels_masks: dict[int, np.ndarray],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Prepare masks for neuropil correction if enabled."""
        eroded_masks = labels_masks
        neuropil_masks_dict = {}
        if (
            extraction_settings.neuropil_inner_radius > 0
            and extraction_settings.neuropil_min_pixels > 0
        ):
            # Get list of masks in order
            sorted_labels = sorted(labels_masks.keys())
            cell_masks = [labels_masks[label] for label in sorted_labels]
            height, width = data.shape[1], data.shape[2]  # assuming data is (t, y, x)
            cell_masks_eroded, neuropil_masks = create_neuropil_from_dilation(
                cell_masks,
                height,
                width,
                inner_neuropil_radius=extraction_settings.neuropil_inner_radius,
                min_neuropil_pixels=extraction_settings.neuropil_min_pixels,
            )
            # Create dicts
            eroded_masks = dict(zip(sorted_labels, cell_masks_eroded))
            neuropil_masks_dict = dict(zip(sorted_labels, neuropil_masks))
        return eroded_masks, neuropil_masks_dict

    def _get_label_mask(
        self, fov_to_analyze: FOV, fov_name: str
    ) -> dict[int, np.ndarray]:
        labels_masks = {}
        for roi in fov_to_analyze.rois:
            if (
                roi.roi_mask
                and roi.roi_mask.coords_y is not None
                and roi.roi_mask.coords_x is not None
                and roi.roi_mask.height is not None
                and roi.roi_mask.width is not None
            ):
                mask_array = coordinates_to_mask(
                    (roi.roi_mask.coords_y, roi.roi_mask.coords_x),
                    (roi.roi_mask.height, roi.roi_mask.width),
                )
                labels_masks[roi.label_value] = mask_array
            else:
                cali_logger.warning(
                    f"ROI {roi.label_value} in {fov_name} has no mask data"
                )

        return labels_masks

    def _compute_roi_dff(
        self,
        data: np.ndarray,
        meta: list[dict],
        fov_name: str,
        extraction_settings: ExtractionSettings,
        label_value: int,
        label_mask: np.ndarray,
        neuropil_mask: np.ndarray | None = None,
        neuropil_correction_factor: float | None = None,
    ) -> _RoiParts | None:
        """Compute raw, corrected, neuropil, and DFF traces before inference."""
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

        return _RoiParts(
            label_value,
            label_mask,
            roi_trace_uncorrected,
            roi_trace if neuropil_trace is not None else None,
            neuropil_trace,
            dff,
            roi_size,
            roi_size_units,
        )

    def _finalize_roi(
        self,
        parts: _RoiParts,
        den_dff: np.ndarray,
        spikes: np.ndarray,
        sn: float,
        analysis_settings: AnalysisSettings | None,
        tot_time_sec: float,
        elapsed_time_list: list[float],
        x_unit: str,
        frame_window: ResolvedFrameWindow,
        *,
        stored_window: ExtractionFrameWindow | None = None,
        inference_run: SpikeInferenceRun | None = None,
        ar_coefficients: list[float] | None = None,
    ) -> tuple[Traces, DataAnalysis | None, bool, bool, float, str] | None:
        """Build trace and optional analysis products from completed inference."""
        if self._check_for_abort_requested():
            return None

        # Create Traces object (extraction product)
        corrected_trace = (
            cast("list[float]", parts.corrected.tolist())
            if parts.corrected is not None
            else None
        )
        if stored_window is None:
            stored_window = ExtractionFrameWindow(
                original_frame_count=frame_window.original_frame_count,
                retained_frame_count=frame_window.retained_frame_count,
                source_start_frame=frame_window.source_start_frame,
                source_start_time_ms=frame_window.source_start_time_ms,
                discarded_duration_ms=frame_window.discarded_duration_ms,
                timing_source=frame_window.timing_source,
            )
        if inference_run is None:
            inference_run = SpikeInferenceRun(
                backend_version=version("oasis-deconv"),
                resolved_device="cpu",
                dtype="float64",
                provenance_source="oasis_inference",
            )
        traces = Traces(
            raw_trace=cast("list[float]", parts.raw.tolist()),
            corrected_trace=corrected_trace,
            neuropil_trace=(
                cast("list[float]", parts.neuropil.tolist())
                if parts.neuropil is not None
                else None
            ),
            dff=cast("list[float]", parts.dff.tolist()),
            den_dff=den_dff.tolist(),
            spike_traces=[
                SpikeTrace(
                    values=spikes.tolist(),
                    valid_stop=len(spikes),
                    noise=sn,
                    ar_coefficients=ar_coefficients,
                    inference_run=inference_run,
                )
            ],
            extraction_frame_window=stored_window,
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
                    stimulated_area_mask, parts.label_mask
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

        return (
            traces,
            data_analysis,
            active,
            stimulated,
            parts.roi_size,
            parts.roi_size_units,
        )

    def _get_fov_name(self, event_key: str, meta: list[dict], p: int) -> str:
        """Retrieve the fov name from metadata.

        Should match the naming used in DetectionRunner to ensure
        analysis can find the FOV created during detection.
        """
        try:
            # Try to get pos_name first (e.g., "B5_0000")
            pos_name = meta[0][event_key].get("pos_name")
            if pos_name:
                return pos_name  # type: ignore[no-any-return]
        except (KeyError, IndexError, AttributeError):
            pass

        # Fallback to constructing from axes
        try:
            well = meta[0][event_key]["axes"]["p"]
            return f"{well}_{p:04d}"
        except (KeyError, IndexError):
            pass

        # Final fallback
        return f"pos_{p}"

    def _get_elapsed_time_ms_list(
        self, meta: list[dict], num_timepoints: int
    ) -> list[float]:
        """Get elapsed time list from metadata."""
        return build_timing_descriptor(meta, num_timepoints).timestamps_ms
