import threading

# ignore deprecation warnings from oasis
import warnings
from collections.abc import Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Callable, cast

import numpy as np
from oasis.functions import deconvolve
from tqdm import tqdm

from cali._constants import (
    EVENT_KEY,
    RUNNER_TIME_KEY,
)
from cali.analysis._util import get_overlap_roi_with_stimulated_area
from cali.logger import cali_logger
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.readers._tiff_collection_reader import TiffCollectionReader
from cali.sqlmodel._model import (
    FOV,
    AnalysisSettings,
    DataAnalysis,
    ExtractionSettings,
    Mask,
    Traces,
)
from cali.util import coordinates_to_mask, mask_to_coordinates

from ._neuropil import create_neuropil_from_dilation
from ._util import calculate_dff

warnings.filterwarnings("ignore", category=FutureWarning)


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

        # Execute analysis in parallel and yield results
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

        global_pos_idx = fov_to_analyze.position_index

        # get the data and metadata for the position
        data, meta = dataset.isel(p=global_pos_idx, metadata=True)
        # get the fov_name name from metadata
        fov_name = self._get_fov_name(EVENT_KEY, meta, global_pos_idx)

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

        # get the elapsed time from the metadata to calculate the total time in seconds
        # assumes data shape is (time, height, width)
        elapsed_time_list = self._get_elapsed_time_ms_list(meta, data.shape[0])

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
            trace_data = self._process_roi_trace(
                data,
                meta,
                fov_name,
                extraction_settings,
                analysis_settings,
                label_value,
                labels_masks[label_value],
                tot_time_sec,
                elapsed_time_list,
                "ms",
                neuropil_masks_dict.get(label_value),
                neuropil_correction_factor,
            )

            # Add traces and analysis to the existing ROI if processing succeeded
            if trace_data is not None:
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

        # Compute FOV-level analysis (correlation/synchrony) if analysis was run
        if analysis_settings is not None and not self._check_for_abort_requested():
            from cali.analysis._fov_analysis import compute_fov_analysis

            fov_analysis = compute_fov_analysis(fov_to_analyze, analysis_settings)
            if fov_analysis is not None:
                # Store in temporary attribute for later commit
                if not hasattr(fov_to_analyze, "_new_fov_analysis"):
                    fov_to_analyze._new_fov_analysis = []
                fov_to_analyze._new_fov_analysis.append(fov_analysis)

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
        px_size = meta[0].get("pixel_size_um", None)
        # Convert to µm² if pixel size is available, otherwise use pixels
        roi_size = roi_size_pixel * (px_size**2) if px_size else roi_size_pixel
        roi_size_units = "µm" if px_size is not None else "pixel"

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
        dff = calculate_dff(roi_trace, window=extraction_settings.dff_window)

        # Check for cancellation after DFF calculation
        if self._check_for_abort_requested():
            return None

        # run OASIS deconvolution on the dff trace
        # compute the decay constant
        tau = extraction_settings.decay_constant
        penalty = 1  # TODO: expose penalty in gui
        g: float | None = None
        if tau > 0.0:
            fs = len(dff) / tot_time_sec  # Sampling frequency (Hz)
            g = np.exp(-1 / (fs * tau))
        # deconvolve the dff trace with adaptive penalty
        dec_dff, spikes, _, _t, _ = deconvolve(dff, penalty=penalty, g=(g,))
        dec_dff = cast("np.ndarray", dec_dff)
        spikes = cast("np.ndarray", spikes)

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
            dec_dff=dec_dff.tolist(),
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
                compute_peak_detection_thresholds,
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

            # Calculate IEI
            iei_ms = calculate_inter_event_intervals(peaks_dec_dff, elapsed_time_list)
            iei = [x / 1000 for x in iei_ms]  # Convert ms to sec

            # Create DataAnalysis object (analysis product)
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

        return (traces, data_analysis, active, stimulated, roi_size, roi_size_units)

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
        elapsed_time_list: list[float] = []

        exposure_ms = cast("float", meta[0].get("exposure_ms", 0.0))

        # if in metadata, get the elapsed time list from RUNNER_TIME_KEY
        if RUNNER_TIME_KEY in meta[0]:  # new metadata format
            for m in meta:
                rt = m[RUNNER_TIME_KEY]
                if rt is not None:
                    elapsed_time_list.append(float(rt))

            # if the elapsed time list is different from the number of
            # timepoints, set it as list of timepoints every exp_time
            if len(elapsed_time_list) != num_timepoints:
                elapsed_time_list = [t * exposure_ms for t in range(num_timepoints)]

        # otherwise use exposure time and number of timepoints to create
        # elapsed time list
        else:
            elapsed_time_list = [t * exposure_ms for t in range(num_timepoints)]
        return elapsed_time_list
