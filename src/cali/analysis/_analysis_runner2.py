import logging
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import tifffile
from oasis.functions import deconvolve
from scipy.signal import find_peaks
from sqlmodel import Session, create_engine, select
from tqdm import tqdm

from cali._constants import (
    EVENT_KEY,
    EVOKED,
    EXCLUDE_AREA_SIZE_THRESHOLD,
    GLOBAL_HEIGHT,
    GLOBAL_SPIKE_THRESHOLD,
    RUNNER_TIME_KEY,
    STIMULATION_AREA_THRESHOLD,
)
from cali.sqlmodel import save_experiment_to_database
from cali.sqlmodel._model import (
    FOV,
    ROI,
    AnalysisResult,
    AnalysisSettings,
    DataAnalysis,
    Experiment,
    Mask,
    Plate,
    Traces,
    Well,
)
from cali.util._util import load_data

from ._util import (
    calculate_dff,
    get_iei,
    get_overlap_roi_with_stimulated_area,
    mask_to_coordinates,
)

if TYPE_CHECKING:
    import useq

    from cali.readers import OMEZarrReader, TensorstoreZarrReader

cali_logger = logging.getLogger("cali_logger")


def commit_result(session: Session, experiment: Experiment, fov_result: FOV) -> None:
    """Commit FOV result to database."""
    # Query for plate ID directly to avoid loading relationships
    plate_statement = (
        select(Plate.id).join(Experiment).where(Experiment.id == experiment.id)
    )
    plate_id_result = session.exec(plate_statement).first()
    if plate_id_result is None:
        cali_logger.error("Experiment plate not initialized")
        return

    # For each FOV, link it to the appropriate well
    well_name = fov_result.name.split("_")[0]  # A1_0000 p -> A1

    # Query for existing well
    well_statement = select(Well).where(
        Well.plate_id == plate_id_result, Well.name == well_name
    )
    well = session.exec(well_statement).first()

    if well is None:
        # Create new well if needed
        row = ord(well_name[0]) - ord("A")
        col = int(well_name[1:]) - 1
        well = Well(
            plate_id=plate_id_result,
            name=well_name,
            row=row,
            column=col,
            fovs=[],
        )
        session.add(well)
        # Get the well ID
        session.flush()

    # Check if FOV already exists for this well (re-analysis case)
    # Now that FOV names are not unique, we need to check by name AND well
    existing_fov_stmt = select(FOV).where(
        FOV.name == fov_result.name, FOV.well_id == well.id
    )
    existing_fov = session.exec(existing_fov_stmt).first()

    if existing_fov:
        # Re-analysis: Delete old ROIs (cascade handles traces, etc.)
        for old_roi in existing_fov.rois:
            session.delete(old_roi)
        session.flush()

        # Update FOV and link to well
        existing_fov.position_index = fov_result.position_index
        existing_fov.fov_number = fov_result.fov_number
        existing_fov.fov_metadata = fov_result.fov_metadata
        existing_fov.well_id = well.id

        # Add new ROIs - detach from fov_result to avoid cascading issues
        # Copy the list to avoid modification during iteration
        rois_to_add = list(fov_result.rois)
        for roi in rois_to_add:
            roi.fov_id = existing_fov.id
            roi.fov = existing_fov  # Explicitly set the relationship
            session.add(roi)
    else:
        # New FOV - link to well and add
        fov_result.well_id = well.id
        session.add(fov_result)

    session.commit()


def exec_(
    analyze: Callable,
    cancel_event: threading.Event,
    global_position_indices: Sequence[int],
    settings: AnalysisSettings,
    experiment: Experiment,
    max_workers: int | None = None,
) -> Iterable[FOV]:
    """Execute analysis in parallel and commit results centrally."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Check for cancellation before submitting futures
        if cancel_event.is_set():
            cali_logger.info("Cancellation requested before starting thread pool")
            return

        futures = (
            executor.submit(analyze, experiment, settings, p)
            for p in global_position_indices
        )

        for future in as_completed(futures):
            # Check for cancellation at the start of each iteration
            if cancel_event.is_set():
                cali_logger.info("Cancellation requested, shutting down executor...")
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
        cali_logger.info("Run Cancelled")


class AnalysisRunner:
    def __init__(self) -> None:
        super().__init__()

        # The data reader
        self._data: TensorstoreZarrReader | OMEZarrReader | None = None
        # Use threading.Event for cancellation control
        self._cancellation_event = threading.Event()

    def run(
        self,
        experiment: Experiment,
        settings: AnalysisSettings,
        global_position_indices: Sequence[int],
        overwrite: bool = False,
        echo: bool = False,
    ) -> None:
        """Run analysis on the given experiment with specified settings."""
        # DATABASE DO NOT EXISTS
        # if database doesn't exist, save experiment to DB for the first time
        if not Path(experiment.db_path).exists():
            save_experiment_to_database(experiment)

        # DATABASE EXISTS
        engine = create_engine(f"sqlite:///{experiment.db_path}", echo=echo)
        with Session(engine) as session:
            # if database does exist but the overwrite flag is True, just overwrite
            if overwrite:
                save_experiment_to_database(experiment, overwrite=True)
            # if database does exist but the the experiment.id is either None or
            # different than the one in the database, raise ValueError.
            else:
                db_exp = cast("Experiment", session.exec(select(Experiment)).first())
                if experiment.id is None or experiment.id != db_exp.id:
                    msg = (
                        "The provided Experiment must have an ID matching the one "
                        f"in the database (ID: {db_exp.id} vs {experiment.id}). Either set the id to "
                        f"{db_exp.id} or set the overwrite flag to `True`."
                    )
                    cali_logger.error(msg)
                    raise ValueError(msg)

            # initialize the analysis runner with data
            self._data = load_data(experiment.data_path)

            # check if settings already exist BEFORE merging
            if settings.id is None:
                # check if identical settings already exist in database
                new_settings_dict = settings.model_dump(exclude={"id", "created_at"})
                try:
                    # AnalysisSettings exists
                    all_settings = session.exec(select(AnalysisSettings)).all()
                    existing_settings = None
                    for candidate in all_settings:
                        candidate_dict = candidate.model_dump(
                            exclude={"id", "created_at"}
                        )
                        if candidate_dict == new_settings_dict:
                            existing_settings = candidate
                            break
                except Exception:
                    # AnalysisSettings doesn't exist yet - this is the first settings
                    existing_settings = None

                # found duplicate - use the existing one
                if existing_settings is not None:
                    settings = existing_settings
                    cali_logger.info(
                        f"♻️ Reusing existing AnalysisSettings ID {settings.id}"
                    )
                # new settings - merge and commit to get an ID
                else:
                    settings = session.merge(settings)
                    session.commit()
                    session.refresh(settings)
                    cali_logger.info(f"⚙️ Created new AnalysisSettings ID {settings.id}")
            else:
                # get all current settings IDs
                all_settings_ids = [
                    s.id for s in session.exec(select(AnalysisSettings)).all()
                ]
                # if settings.id is new, merge and commit
                if settings.id not in all_settings_ids:
                    settings = session.merge(settings)
                    session.commit()
                    session.refresh(settings)
                    cali_logger.info(f"⚙️ Created new AnalysisSettings ID {settings.id}")
                else:
                    # settings already has an ID - just merge to reattach
                    settings = session.merge(settings)
                    cali_logger.info(
                        f"♻️ Reusing existing AnalysisSettings ID {settings.id}"
                    )

            # ensure experiment has an ID
            if experiment.id is None:
                msg = "Experiment must have an ID before running analysis"
                raise ValueError(msg)

            # track positions processed
            positions_processed = []

            # execute analysis in parallel
            for result in exec_(
                analyze=self._analyze_position,
                cancel_event=self._cancellation_event,
                global_position_indices=global_position_indices,
                settings=settings,
                experiment=experiment,
                max_workers=settings.threads,
            ):
                commit_result(session, experiment, result)
                positions_processed.append(result.position_index)

            # once the analysis is complete, create or update AnalysisResult
            if positions_processed:
                assert settings.id is not None  # should never ne None here

                # query for existing AnalysisResult with same experiment + settings
                existing_result_stmt = select(AnalysisResult).where(
                    AnalysisResult.experiment == experiment.id,
                    AnalysisResult.analysis_settings == settings.id,
                )
                existing_result = session.exec(existing_result_stmt).first()

                if existing_result:
                    # check if positions match - if so, overwrite; if not, add new
                    if set(existing_result.positions_analyzed or []) != set(
                        positions_processed
                    ):
                        # different positions - this is a new analysis run
                        new_result = AnalysisResult(
                            experiment=experiment.id,
                            analysis_settings=settings.id,
                            positions_analyzed=positions_processed,
                        )
                        session.add(new_result)
                    # else is the same positions - just update the timestamp implicitly
                    # (no changes needed, data already committed)
                else:
                    # no existing result - create new
                    new_result = AnalysisResult(
                        experiment=experiment.id,
                        analysis_settings=settings.id,
                        positions_analyzed=positions_processed,
                    )
                    session.add(new_result)
                    cali_logger.info("✅ Created new AnalysisResult.")

                session.commit()

    def _analyze_position(
        self, experiment: Experiment, settings: AnalysisSettings, global_pos_idx: int
    ) -> FOV | None:
        """Extract the roi traces for the given position and return result objects.

        Returns a list of FOV objects with all their nested relationships
        (ROIs, Traces, DataAnalysis, Masks) ready to be committed.
        """
        return _extract_trace_data_per_position(
            self, experiment, settings, global_pos_idx
        )

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_event.is_set()


# These are module-level functions that work with the OriginalAnalysisRunner
def _extract_trace_data_per_position(
    runner: AnalysisRunner,
    experiment: Experiment,
    settings: AnalysisSettings,
    global_pos_idx: int,
) -> FOV | None:
    """Extract trace data for a position and return FOV objects (not committed).

    Returns a list containing a single FOV with all its ROIs, Traces, DataAnalysis,
    and Masks ready to be committed to the database.
    """
    # if runner._data is None or runner._check_for_abort_requested():
    if runner._data is None or runner._check_for_abort_requested():
        return None

    # get the data and metadata for the position
    data, meta = runner._data.isel(p=global_pos_idx, metadata=True)

    # get the fov_name name from metadata
    fov_name = _get_fov_name(EVENT_KEY, meta, global_pos_idx)

    # get the labels file for the position
    labels_path = _get_labels_file_for_position(experiment, fov_name, global_pos_idx)
    if labels_path is None:
        return None

    # open the labels file and create masks for each label
    labels = tifffile.imread(labels_path)

    # Check for cancellation after file I/O operation
    if runner._check_for_abort_requested():
        return None

    # { roi_id -> np.ndarray mask }
    labels_masks = _create_label_masks_dict(labels)
    sequence = cast("useq.MDASequence", runner._data.sequence)

    # Check for cancellation after loading and processing labels
    if runner._check_for_abort_requested():
        return None

    # Prepare masks for neuropil correction if enabled
    from ._util import create_neuropil_from_dilation

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

    # get the exposure time from the metadata
    exp_time = meta[0][EVENT_KEY].get("exposure", 0.0)
    # get timepoints
    timepoints = sequence.sizes["t"]
    # get the elapsed time from the metadata to calculate the total time in seconds
    elapsed_time_list = _get_elapsed_time_list(meta)
    # if the elapsed time is not available or for any reason is different from
    # the number of timepoints, set it as list of timepoints every exp_time
    if len(elapsed_time_list) != timepoints:
        elapsed_time_list = [i * exp_time for i in range(timepoints)]
    # get the total time in seconds for the recording
    tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0]) / 1000

    # check if it is an evoked activity experiment
    evoked_experiment = experiment.experiment_type == EVOKED

    # Create the FOV object (not yet committed)
    fov_name.split("_")[0]
    fov = FOV(
        name=fov_name,
        position_index=global_pos_idx,
        fov_number=global_pos_idx,
        rois=[],
    )

    # >>>> HERE is the big loop over roi mask, calling _process_roi_trace

    msg = f"Extracting Traces Data from {fov_name}."
    cali_logger.info(msg)
    for label_value, _label_mask in tqdm(labels_masks.items(), desc=msg):
        if runner._check_for_abort_requested():
            cali_logger.info(f"Cancellation requested during processing of {fov_name}")
            break

        # Process each ROI and get the result objects (not committed)
        roi_result = _process_roi_trace(
            runner,
            data,
            meta,
            fov_name,
            settings,
            label_value,
            eroded_masks[label_value],
            tot_time_sec,
            evoked_experiment,
            elapsed_time_list,
            neuropil_masks_dict.get(label_value),
            (
                settings.neuropil_correction_factor
                if (
                    settings.neuropil_inner_radius > 0
                    and settings.neuropil_min_pixels > 0
                )
                else None
            ),
        )

        # Add the ROI to the FOV if processing succeeded
        if roi_result:
            fov.rois.append(roi_result)

    # Return the FOV with all its ROIs (will be committed by caller)
    return fov


def _process_roi_trace(
    runner: AnalysisRunner,
    data: np.ndarray,
    meta: list[dict],
    fov_name: str,
    settings: AnalysisSettings,
    label_value: int,
    label_mask: np.ndarray,
    tot_time_sec: float,
    evoked_exp: bool,
    elapsed_time_list: list[float],
    neuropil_mask: np.ndarray | None = None,
    neuropil_correction_factor: float | None = None,
) -> ROI | None:
    """Process individual ROI trace and return ROI object (not committed).

    Returns an ROI object with Traces, DataAnalysis, and Masks ready to commit,
    or None if processing fails or ROI should be excluded.
    """
    # Early exit if cancellation is requested
    if runner._check_for_abort_requested():
        return None

    # get the data for the current label
    masked_data = data[:, label_mask]

    # get the size of the roi in µm or px if µm is not available
    roi_size_pixel = masked_data.shape[1]  # area
    px_keys = ["pixel_size_um", "PixelSizeUm"]
    px_size = None
    for key in px_keys:
        px_size = meta[0].get(key, None)
        if px_size:
            break
    # calculate the size of the roi in µm if px_size is available or not 0,
    # otherwise use the size is in pixels
    roi_size = roi_size_pixel * (px_size**2) if px_size else roi_size_pixel

    # exclude small rois
    if px_size and roi_size < EXCLUDE_AREA_SIZE_THRESHOLD:
        return None

    # check if the roi is stimulated
    roi_stimulation_overlap_ratio = 0.0
    stimulated_area_mask = settings.stimulated_mask_area()
    if evoked_exp and stimulated_area_mask is not None:
        roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
            stimulated_area_mask, label_mask
        )

    # compute the mean for each frame
    roi_trace_uncorrected: np.ndarray = masked_data.mean(axis=1)
    win = settings.dff_window

    # Check for cancellation before DFF calculation
    if runner._check_for_abort_requested():
        return None

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
    dff = calculate_dff(roi_trace, window=win, plot=False)

    # Check for cancellation after DFF calculation
    if runner._check_for_abort_requested():
        return None

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
    if runner._check_for_abort_requested():
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
    if runner._check_for_abort_requested():
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
    if runner._check_for_abort_requested():
        return None

    # find peaks in the deconvolved trace
    peaks_dec_dff, _ = find_peaks(
        dec_dff,
        prominence=peaks_prominence_dec_dff,
        height=peaks_height_dec_dff,
        distance=min_distance_frames,
    )
    peaks_dec_dff = cast("np.ndarray", peaks_dec_dff)

    # Check for cancellation after peak finding
    if runner._check_for_abort_requested():
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
    if runner._check_for_abort_requested():
        return None

    # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
    iei = get_iei(peaks_dec_dff, elapsed_time_list)

    # get mask coords and shape for the ROI
    mask_coords, mask_shape = mask_to_coordinates(label_mask)

    # get neuropil mask coords and shape if neuropil mask exists
    neuropil_mask_coords = neuropil_mask_shape = None
    if neuropil_mask is not None:
        neuropil_mask_coords, neuropil_mask_shape = mask_to_coordinates(neuropil_mask)

    # Create mask objects
    roi_mask = Mask(
        coords_y=mask_coords[0],
        coords_x=mask_coords[1],
        height=mask_shape[0],
        width=mask_shape[1],
        mask_type="roi",
    )

    neuropil_mask_obj = None
    if neuropil_mask_coords and neuropil_mask_shape:
        neuropil_mask_obj = Mask(
            coords_y=neuropil_mask_coords[0],
            coords_x=neuropil_mask_coords[1],
            height=neuropil_mask_shape[0],
            width=neuropil_mask_shape[1],
            mask_type="neuropil",
        )

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

    # Create ROI object with all relationships
    roi = ROI(
        label_value=label_value,
        active=len(peaks_dec_dff) > 0,
        stimulated=is_roi_stimulated,
        roi_mask=roi_mask,
        neuropil_mask=neuropil_mask_obj,
        traces=traces,
        data_analysis=data_analysis,
    )
    # Use foreign key to avoid trying to INSERT the experiment
    roi.analysis_settings_id = settings.id

    return roi


def _get_fov_name(event_key: str, meta: list[dict], p: int) -> str:
    """Retrieve the fov name from metadata."""
    # the "Event" key was used in the old metadata format
    pos_name = meta[0].get(event_key, {}).get("pos_name", f"pos_{str(p).zfill(4)}")
    return f"{pos_name}_p{p}"


def _get_labels_file_for_position(
    experiment: Experiment, fov: str, p: int
) -> str | None:
    """Retrieve the labels file for the given position."""
    # if the fov name does not end with "_p{p}", add it
    labels_name = f"{fov}.tif" if fov.endswith(f"_p{p}") else f"{fov}_p{p}.tif"
    labels_path = _get_labels_file(experiment, labels_name)
    if labels_path is None:
        cali_logger.error("No labels found for %s!", labels_name)
    return labels_path


def _get_labels_file(experiment: Experiment, label_name: str) -> str | None:
    """Get the labels file for the given name."""
    if (labels_path := experiment.labels_path) is None:
        return None
    for label_file in Path(labels_path).glob("*.tif"):
        if label_file.name.endswith(label_name):
            return str(label_file)
    return None


def _create_label_masks_dict(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Create masks for each label in the labels image."""
    # get the range of labels and remove the background (0)
    labels_range = np.unique(labels[labels != 0])
    # Convert numpy int to Python int to avoid bytes serialization issues
    return {int(label_value): (labels == label_value) for label_value in labels_range}


def _get_elapsed_time_list(meta: list[dict]) -> list[float]:
    """Get elapsed time list from metadata."""
    elapsed_time_list: list[float] = []
    # get the elapsed time for each timepoint to calculate tot_time_sec
    if RUNNER_TIME_KEY in meta[0]:  # new metadata format
        for m in meta:
            rt = m[RUNNER_TIME_KEY]
            if rt is not None:
                elapsed_time_list.append(float(rt))
    return elapsed_time_list
