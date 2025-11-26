import threading
from collections.abc import Generator, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, cast

import numpy as np
from oasis.functions import deconvolve
from scipy.signal import find_peaks
from tqdm import tqdm

from cali._constants import (
    EVENT_KEY,
    GLOBAL_HEIGHT,
    GLOBAL_SPIKE_THRESHOLD,
    RUNNER_TIME_KEY,
    STIMULATION_AREA_THRESHOLD,
)
from cali.logger import cali_logger
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.sqlmodel._model import (
    FOV,
    AnalysisSettings,
    DataAnalysis,
    Mask,
    Traces,
)
from cali.util import coordinates_to_mask, load_data, mask_to_coordinates

from ._neuropil import create_neuropil_from_dilation
from ._util import (
    calculate_dff,
    get_iei,
    get_overlap_roi_with_stimulated_area,
)


def exec_(
    analyze: Callable,
    dataset: TensorstoreZarrReader | OMEZarrReader,
    cancel_event: threading.Event,
    global_position_indices: Sequence[int],
    settings: AnalysisSettings,
    fovs_with_rois: list[FOV],
    max_workers: int | None = None,
) -> Iterable[FOV]:
    """Execute analysis in parallel and yield FOV results."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Check for cancellation before submitting futures
        if cancel_event.is_set():
            cali_logger.info("🚮 Cancellation requested before starting thread pool")
            return

        futures = (
            executor.submit(
                analyze,
                dataset,
                settings,
                p,
                fovs_with_rois,
            )
            for p in global_position_indices
        )

        for future in as_completed(futures):
            # Check for cancellation at the start of each iteration
            if cancel_event.is_set():
                cali_logger.info("🚮 Cancellation requested, shutting down executor...")
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

    # Check if cancelled before finishing
    if cancel_event.is_set():
        cali_logger.info("❌ Run Cancelled")


class AnalysisRunner:
    def __init__(self) -> None:
        super().__init__()

        # Use threading.Event for cancellation control
        self._cancellation_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the analysis process."""
        cali_logger.info("🗑️ Cancellation requested...")
        self._cancellation_event.set()

    def run(
        self,
        dataset: str | Path | TensorstoreZarrReader | OMEZarrReader,
        settings: AnalysisSettings,
        fovs_with_rois: list[FOV],
        global_position_indices: Sequence[int],
    ) -> Generator[FOV, None, None]:
        """Run analysis and yield FOV results with traces and analysis data.

        This method performs pure computation - it takes FOVs with ROIs and adds
        traces and analysis data to them. It does not interact with the database.

        Parameters
        ----------
        dataset : str | Path | TensorstoreZarrReader | OMEZarrReader
            Path to imaging data (zarr store) or a data reader instance
        settings : AnalysisSettings
            Analysis parameters
        fovs_with_rois : list[FOV]
            FOVs with ROIs to analyze. These typically come from DetectionRunner
            or are loaded from the database by the caller.
        global_position_indices : Sequence[int]
            Position indices to analyze

        Yields
        ------
        FOV
            FOV objects with ROIs containing traces and analysis data,
            ready to be saved to database

        Raises
        ------
        ValueError
            If no ROI masks are found in the provided FOVs
        """
        # Reset cancellation event
        self._cancellation_event.clear()

        # Load data
        if isinstance(dataset, (str, Path)):
            dataset = load_data(dataset)
        else:
            assert isinstance(dataset, (TensorstoreZarrReader, OMEZarrReader)), (
                "Data must be a TensorstoreZarrReader or OMEZarrReader instance."
            )

        # Execute analysis in parallel and yield results
        for fov_result in exec_(
            analyze=self._analyze_position,
            dataset=dataset,
            cancel_event=self._cancellation_event,
            global_position_indices=global_position_indices,
            settings=settings,
            fovs_with_rois=fovs_with_rois,
            max_workers=settings.threads,
        ):
            if fov_result is not None:
                yield fov_result

        cali_logger.info("✅ Analysis complete!")

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_event.is_set()

    def _analyze_position(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader,
        settings: AnalysisSettings,
        global_pos_idx: int,
        fovs_with_rois: list[FOV],
    ) -> FOV | None:
        """Extract the roi traces for the given position and return result objects.

        Returns a FOV object with all its nested relationships
        (ROIs, Traces, DataAnalysis, Masks) ready to be committed.
        """
        return self._extract_trace_data_per_position(
            dataset,
            settings,
            global_pos_idx,
            fovs_with_rois,
        )

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

    # These are module-level functions that work with the OriginalAnalysisRunner
    def _extract_trace_data_per_position(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader,
        settings: AnalysisSettings,
        global_pos_idx: int,
        fovs_with_rois: list[FOV],
    ) -> FOV | None:
        """Extract trace data for a position and return FOV objects (not committed).

        Returns a FOV with all its ROIs, Traces, DataAnalysis, and Masks
        ready to be committed to the database.
        """
        # if runner._data is None or runner._check_for_abort_requested():
        if self._check_for_abort_requested():
            return None

        # get the data and metadata for the position
        data, meta = dataset.isel(p=global_pos_idx, metadata=True)
        # get the fov_name name from metadata
        fov_name = _get_fov_name(EVENT_KEY, meta, global_pos_idx)

        # Find the FOV with matching position index in the provided list
        fov_to_analyze = self._get_fov_to_analyze(global_pos_idx, fovs_with_rois)

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
            settings, data, labels_masks
        )

        # get the elapsed time from the metadata to calculate the total time in seconds
        # assumes data shape is (time, height, width)
        exposure_ms = None  # TODO: expose frame rate/exp time in the gui
        elapsed_time_list = _get_elapsed_time_list(meta, exposure_ms, data.shape[0])

        # get the total time in seconds for the recording
        tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0]) / 1000

        # Use the existing FOV from detection (don't create a new one)
        # We'll add traces to the existing ROIs
        msg = f"📈 Extracting Traces Data from {fov_name}."
        cali_logger.info(msg)

        # Create a map of label_value -> ROI for quick lookup
        roi_map = {roi.label_value: roi for roi in fov_to_analyze.rois}

        for label_value in tqdm(labels_masks.keys(), desc=msg):
            if self._check_for_abort_requested():
                cali_logger.info(
                    f"🚮 Cancellation requested during processing of {fov_name}"
                )
                break

            # Get the existing ROI from detection
            existing_roi = roi_map.get(label_value)
            if existing_roi is None:
                cali_logger.warning(
                    f"No ROI found with label_value={label_value} in {fov_name}"
                )
                continue

            # Process the trace and get Traces + DataAnalysis objects
            neuropil_correction_factor = (
                settings.neuropil_correction_factor
                if (
                    settings.neuropil_inner_radius > 0
                    and settings.neuropil_min_pixels > 0
                )
                else None
            )
            trace_data = self._process_roi_trace(
                data,
                meta,
                fov_name,
                settings,
                label_value,
                labels_masks[label_value],
                tot_time_sec,
                elapsed_time_list,
                neuropil_masks_dict.get(label_value),
                neuropil_correction_factor,
            )

            # Add traces and analysis to the existing ROI if processing succeeded
            if trace_data is not None:
                traces, data_analysis, active, stimulated = trace_data

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

                # Append directly to relationships (eagerly loaded by caller)
                existing_roi.traces_history.append(traces)
                existing_roi.data_analysis_history.append(data_analysis)
                existing_roi.active = active
                existing_roi.stimulated = stimulated

        # Return the FOV with updated ROIs (will be committed by caller)
        return fov_to_analyze

    def _prepare_neuropil_masks(
        self,
        settings: AnalysisSettings,
        data: np.ndarray,
        labels_masks: dict[int, np.ndarray],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Prepare masks for neuropil correction if enabled."""
        eroded_masks = labels_masks
        neuropil_masks_dict = {}
        if settings.neuropil_inner_radius > 0 and settings.neuropil_min_pixels > 0:
            # Get list of masks in order
            sorted_labels = sorted(labels_masks.keys())
            cell_masks = [labels_masks[label] for label in sorted_labels]
            height, width = data.shape[1], data.shape[2]  # assuming data is (t, y, x)
            cell_masks_eroded, neuropil_masks = create_neuropil_from_dilation(
                cell_masks,
                height,
                width,
                inner_neuropil_radius=settings.neuropil_inner_radius,
                min_neuropil_pixels=settings.neuropil_min_pixels,
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
        settings: AnalysisSettings,
        label_value: int,
        label_mask: np.ndarray,
        tot_time_sec: float,
        elapsed_time_list: list[float],
        neuropil_mask: np.ndarray | None = None,
        neuropil_correction_factor: float | None = None,
    ) -> tuple[Traces, DataAnalysis, bool, bool] | None:
        """Process individual ROI trace and return trace data.

        Returns a tuple of (Traces, DataAnalysis, active, stimulated) ready to add
        to an existing ROI, or None if processing fails or ROI should be excluded.
        """
        # Early exit if cancellation is requested
        if self._check_for_abort_requested():
            return None

        # get the data for the current label
        masked_data = data[:, label_mask]

        # get the size of the roi in µm or px if µm is not available
        roi_size_pixel = masked_data.shape[1]  # area
        px_size = meta[0].get("pixel_size_um", None)
        # calculate the size of the roi in µm if px_size is available or not 0,
        # otherwise use the size is in pixels
        roi_size = roi_size_pixel * (px_size**2) if px_size else roi_size_pixel

        # # exclude small rois
        # if px_size and roi_size < EXCLUDE_AREA_SIZE_THRESHOLD:
        #     return None

        # check if the roi is stimulated
        roi_stimulation_overlap_ratio = 0.0
        stimulated_area_mask = settings.stimulated_mask_area()
        if stimulated_area_mask is not None:
            roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
                stimulated_area_mask, label_mask
            )

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
        dff = calculate_dff(roi_trace, window=settings.dff_window, plot=False)

        # Check for cancellation after DFF calculation
        if self._check_for_abort_requested():
            return None

        # run OASIS deconvolution on the dff trace
        # compute the decay constant
        tau = settings.decay_constant
        g: float | None = None
        if tau > 0.0:
            fs = len(dff) / tot_time_sec  # Sampling frequency (Hz)
            g = np.exp(-1 / (fs * tau))
        # deconvolve the dff trace with adaptive penalty
        dec_dff, spikes, _, _t, _ = deconvolve(dff, penalty=1, g=(g,))
        dec_dff = cast("np.ndarray", dec_dff)
        spikes = cast("np.ndarray", spikes)

        # Check for cancellation after deconvolution
        if self._check_for_abort_requested():
            return None

        # Use the spike threshold widget to get the spike detection threshold
        spike_threshold_value = settings.spike_threshold_value
        spike_threshold_mode = settings.spike_threshold_mode

        if spike_threshold_mode == GLOBAL_SPIKE_THRESHOLD:
            spike_detection_threshold = spike_threshold_value
        else:  # MULTIPLIER
            # for spike amp use percentile-based approach to determine noise level
            non_zero_spikes = spikes[spikes > 0]
            # need sufficient data for reliable percentile
            if len(non_zero_spikes) > 5:
                spike_noise_reference = float(np.percentile(non_zero_spikes, 5))
            else:
                cali_logger.warning(
                    "Not enough data to determine spike noise reference "
                    "(< 5 non-zero spikes), using fallback value of 0.01."
                )
                spike_noise_reference = 0.01  # fallback value if not enough data
            spike_detection_threshold = spike_noise_reference * spike_threshold_value

        # Get noise level from the ΔF/F0 trace using Median Absolute Deviation (MAD)
        noise_level_dec_dff = float(
            np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745
        )

        # Check for cancellation after noise level calculation
        if self._check_for_abort_requested():
            return None

        # Set prominence threshold (how much peaks must stand out from surroundings)
        # Use a fraction of noise level to be less restrictive than height threshold
        prom_multiplier = settings.peaks_prominence_multiplier
        peaks_prominence_dec_dff: float = noise_level_dec_dff * prom_multiplier

        # use the peaks height widget to get the height threshold
        # if the mode is GLOBAL_HEIGHT, use the value directly, otherwise
        # use the value as a multiplier of the noise level
        peaks_height_value = settings.peaks_height_value
        peaks_height_mode = settings.peaks_height_mode
        if peaks_height_mode == GLOBAL_HEIGHT:
            peaks_height_dec_dff = peaks_height_value
        else:  # MULTIPLIER
            peaks_height_dec_dff = noise_level_dec_dff * peaks_height_value

        # Get minimum distance between peaks from user-specified value
        min_distance_frames = settings.peaks_distance

        # Check for cancellation before peak finding
        if self._check_for_abort_requested():
            return None

        # find peaks in the deconvolved trace
        peaks_dec_dff, _ = find_peaks(
            dec_dff,
            prominence=peaks_prominence_dec_dff,
            height=peaks_height_dec_dff,
            distance=min_distance_frames,
        )
        peaks_dec_dff = cast("np.ndarray", peaks_dec_dff)

        # TODO: find peaks also in spikes traces

        # Check for cancellation after peak finding
        if self._check_for_abort_requested():
            return None

        # get the amplitudes of the peaks in the dec_dff trace
        peaks_amplitudes_dec_dff = [float(dec_dff[p]) for p in peaks_dec_dff]

        # check if the roi is stimulated
        is_roi_stimulated = roi_stimulation_overlap_ratio > STIMULATION_AREA_THRESHOLD

        # calculate the frequency of the peaks in the dec_dff trace
        frequency = (
            len(peaks_dec_dff) / tot_time_sec
            if tot_time_sec and len(peaks_dec_dff) > 0
            else None
        )

        # Check for cancellation before final data processing and storage
        if self._check_for_abort_requested():
            return None

        # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
        iei = get_iei(peaks_dec_dff, elapsed_time_list)

        # Create Traces object
        traces = Traces(
            raw_trace=cast("list[float]", roi_trace_uncorrected.tolist()),
            corrected_trace=cast("list[float]", roi_trace.tolist()),
            neuropil_trace=(
                cast("list[float]", neuropil_trace.tolist())
                if neuropil_trace is not None
                else None
            ),
            dff=cast("list[float]", dff.tolist()),
            dec_dff=dec_dff.tolist(),
            x_axis=elapsed_time_list,
        )

        # Create DataAnalysis object
        data_analysis = DataAnalysis(
            cell_size=roi_size,
            cell_size_units="µm" if px_size is not None else "pixel",
            total_recording_time_sec=tot_time_sec,
            dec_dff_frequency=frequency,
            peaks_dec_dff=peaks_dec_dff.tolist(),
            peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
            iei=iei,
            inferred_spikes=spikes.tolist(),
            peaks_prominence_dec_dff=peaks_prominence_dec_dff,
            peaks_height_dec_dff=peaks_height_dec_dff,
            inferred_spikes_threshold=spike_detection_threshold,
        )

        # Return trace data to be added to existing ROI
        active = len(peaks_dec_dff) > 0
        stimulated = is_roi_stimulated

        return (traces, data_analysis, active, stimulated)


def _get_fov_name(event_key: str, meta: list[dict], p: int) -> str:
    """Retrieve the fov name from metadata.

    Should match the naming used in DetectionRunner to ensure
    analysis can find the FOV created during detection.
    """
    try:
        # Try to get pos_name first (e.g., "B5_0000")
        pos_name = meta[0][event_key].get("pos_name")
        if pos_name:
            return pos_name
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


def _get_elapsed_time_list(
    meta: list[dict], exposure_ms: float | None, num_timepoints: int
) -> list[float]:
    """Get elapsed time list from metadata."""
    elapsed_time_list: list[float] = []

    # from metadata get the exposure time in ms
    if exposure_ms is None:
        exposure_ms = cast("float", meta[0].get("exposure_ms", 0.0))

    # if in metadata, get the elapsed time list from RUNNER_TIME_KEY
    if RUNNER_TIME_KEY in meta[0]:  # new metadata format
        for m in meta:
            rt = m[RUNNER_TIME_KEY]
            if rt is not None:
                elapsed_time_list.append(float(rt))

        # if the elapsed time list is different from the number of timepoints, set it
        # as list of timepoints every exp_time
        if len(elapsed_time_list) != num_timepoints:
            elapsed_time_list = [t * exposure_ms for t in range(num_timepoints)]

    # otherwise use exposure time and number of timepoints to create elapsed time list
    else:
        elapsed_time_list = [t * exposure_ms for t in range(num_timepoints)]
    return elapsed_time_list
