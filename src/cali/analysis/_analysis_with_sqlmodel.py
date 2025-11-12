from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
from oasis.functions import deconvolve
from psygnal import Signal
from scipy.signal import find_peaks
from sqlmodel import Session, create_engine
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
from cali.logger import cali_logger
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    DataAnalysis,
    Experiment,
    Mask,
    Plate,
    Traces,
    Well,
    save_experiment_to_database,
)
from cali.util._util import load_data

from ._util import (
    calculate_dff,
    coordinates_to_mask,
    create_neuropil_from_dilation,
    get_iei,
    get_overlap_roi_with_stimulated_area,
    mask_to_coordinates,
)

if TYPE_CHECKING:
    import useq
    from sqlalchemy.engine import Engine

    from cali.readers import OMEZarrReader, TensorstoreZarrReader


class AnalysisRunner:
    analysisInfo: Signal = Signal(str, str)  # message, type
    progressUpdated: Signal = Signal(str)  # analysis progress updates

    def __init__(self) -> None:
        super().__init__()

        # Instead of keeping the full experiment in memory, store engine and ID
        self._engine: Engine | None = None

        # The data reader
        self._data: TensorstoreZarrReader | OMEZarrReader | None = None

        # Use threading.Event for cancellation control
        self._cancellation_event = threading.Event()

        # list to store the failed labels if they will not be found during the
        # analysis. used to show at the end of the analysis to the user which labels
        # are failed to be found.
        self._failed_labels: list[str] = []

    # PUBLIC METHODS -----------------------------------------------------------------

    def experiment(self) -> Experiment | None:
        """Load and return the current experiment from database.

        This follows SQLModel best practices by loading from the database
        when needed rather than keeping objects in memory.
        """
        if self._engine is None:
            return None

        from sqlmodel import Session, select

        with Session(self._engine) as session:
            exp = session.exec(select(Experiment)).first()
            if exp:
                # Force load relationships while session is active
                if exp.plate:
                    _ = len(exp.plate.wells)
                    for well in exp.plate.wells:
                        _ = len(well.fovs)
                        for fov in well.fovs:
                            _ = len(fov.rois)
                            # Force load ROI traces and data analysis
                            for roi in fov.rois:
                                _ = roi.traces
                                _ = roi.data_analysis
                _ = exp.analysis_settings
                session.expunge(exp)
            return exp

    def set_experiment(self, experiment: Experiment) -> None:
        """Set the experiment and initialize database connection.

        This saves the experiment to database and stores the engine + ID
        following SQLModel best practices.
        """
        if experiment.analysis_path is None or experiment.database_name is None:
            # TODO: send error signal to gui
            cali_logger.error(
                "Experiment must have BOTH analysis_path and database_name set!"
            )
            return

        # save the provided experiment to database
        save_experiment_to_database(experiment, overwrite=True)
        # this should never be None here
        assert experiment.analysis_path is not None
        assert experiment.database_name is not None
        assert experiment.id is not None

        # Store engine instead of the full object
        db_path = Path(experiment.analysis_path) / experiment.database_name
        self._engine = create_engine(f"sqlite:///{db_path}")

        # if experiments has data_path, try to load the data
        if experiment.data_path:
            self._data = load_data(experiment.data_path)

    def data(self) -> TensorstoreZarrReader | OMEZarrReader | None:
        """Return the current data reader."""
        return self._data

    def set_data(
        self, data: TensorstoreZarrReader | OMEZarrReader | str | Path
    ) -> None:
        """Set the data reader or load data from path."""
        self._data = load_data(data) if isinstance(data, (str, Path)) else data

        # update experiment with the provided data path if different
        path = data if isinstance(data, (str, Path)) else data.path

        if self._engine is None:
            cali_logger.error(
                "No database engine found. Cannot update data path. "
                "Please set an experiment first with set_experiment(Experiment)."
            )
            return

        from sqlmodel import Session, select

        with Session(self._engine) as session:
            exp = session.exec(select(Experiment)).first()
            if exp and str(path) != exp.data_path:
                exp.data_path = str(path)
                session.add(exp)
                session.commit()
                cali_logger.info(
                    f"💾 Experiment data path updated in database at "
                    f"{exp.analysis_path}/{exp.database_name}"
                )

        if self._data is None:
            cali_logger.error(f"Failed to load data from the provided path: {path}.")

    def settings(self) -> AnalysisSettings | None:
        """Get current analysis settings from database."""
        if self._engine is None:
            return None

        from sqlmodel import Session, select

        with Session(self._engine) as session:
            exp = session.exec(select(Experiment)).first()
            if exp:
                settings = exp.analysis_settings
                if settings:
                    session.expunge(settings)
                return settings
        return None

    def update_settings(self, settings: AnalysisSettings) -> None:
        """Update the analysis settings following SQLModel best practices.

        Uses explicit session management and proper deletion patterns.
        """
        if self._engine is None:
            cali_logger.error(
                "No database engine found. Cannot update settings. "
                "Please set an experiment first with set_experiment(Experiment)."
            )
            return

        from sqlmodel import Session, select

        with Session(self._engine) as session:
            # Get the current experiment
            statement = select(Experiment)
            exp = session.exec(statement).first()

            if not exp:
                cali_logger.error("Experiment not found.")
                return

            # Delete existing settings if present
            if exp.analysis_settings:
                session.delete(exp.analysis_settings)

            # Add or merge the new settings
            exp.analysis_settings = settings
            session.add(exp)
            session.add(settings)
            session.commit()

            cali_logger.info(
                f"💾 Analysis settings updated in database at "
                f"{exp.analysis_path}/{exp.database_name}"
            )

    def set_positions_to_analyze(self, positions: list[int]) -> None:
        """Set which positions should be analyzed.

        Updates the experiment's positions_analyzed list in the database.
        Validates that the positions exist in the loaded data.

        Parameters
        ----------
        positions : list[int]
            List of position indices to analyze (e.g., [0, 1, 4, 5])

        Raises
        ------
        ValueError
            If any position index is invalid (negative or beyond available positions)

        Example
        -------
        >>> analysis = AnalysisRunner()
        >>> analysis.set_experiment(exp)
        >>> analysis.set_positions_to_analyze([0, 2, 4])  # Analyze only these positions
        >>> analysis.run()
        """
        if self._engine is None:
            cali_logger.error(
                "No database engine found. Cannot set positions. "
                "Please set an experiment first with set_experiment(Experiment)."
            )
            return

        # Validate positions against loaded data
        if self._data is not None:
            # Get total number of positions from data
            if hasattr(self._data, "sequence") and self._data.sequence is not None:
                sequence = self._data.sequence
                if hasattr(sequence, "stage_positions"):
                    total_positions = len(sequence.stage_positions)
                else:
                    # Fallback: try to get from data shape
                    total_positions = getattr(self._data, "sizes", {}).get("p", None)
            else:
                total_positions = None

            if total_positions is not None:
                # Check for invalid positions
                invalid_positions = [
                    p for p in positions if p < 0 or p >= total_positions
                ]
                if invalid_positions:
                    error_msg = (
                        f"Invalid position indices: {invalid_positions}. "
                        f"Data has {total_positions} positions "
                        f"(0 to {total_positions - 1})."
                    )
                    cali_logger.error(error_msg)
                    return

                cali_logger.info(
                    f"✅ Validated positions {positions} "
                    f"(data has {total_positions} positions total)"
                )
            else:
                cali_logger.warning(
                    "Could not determine total number of positions from data. "
                    "Skipping validation."
                )
        else:
            cali_logger.warning(
                "No data loaded. Cannot validate positions. "
                "Make sure to call set_experiment() or set_data() first."
            )

        from sqlmodel import Session, select

        with Session(self._engine) as session:
            # Get the current experiment
            statement = select(Experiment)
            exp = session.exec(statement).first()

            if not exp:
                cali_logger.error("Experiment not found.")
                return

            # Update positions to analyze
            exp.positions_analyzed = positions
            session.add(exp)
            session.commit()

            cali_logger.info(
                f"💾 Positions to analyze updated: {positions} in database at "
                f"{exp.analysis_path}/{exp.database_name}"
            )

    def run(self) -> None:
        """Run the analysis."""
        cali_logger.info("Starting Analysis...")

        exp = self.experiment()
        if exp is None:
            cali_logger.error("No Experiment set for analysis.")
            self.analysisInfo.emit("No Experiment set for analysis.", "error")
            return
        if self._data is None:
            cali_logger.error("No Data set for analysis.")
            self.analysisInfo.emit("No Data set for analysis.", "error")
            return

        if exp.analysis_settings is None:
            cali_logger.error("No AnalysisSettings found in Experiment.")
            self.analysisInfo.emit("No AnalysisSettings found in Experiment.", "error")
            return

        # Reset cancellation event
        self._cancellation_event.clear()

        self._extract_traces_data()

    def cancel(self) -> None:
        """Cancel the running analysis."""
        cali_logger.info("Cancellation requested...")
        self._cancellation_event.set()

    def clear_analysis_results(self) -> None:
        """Clear all analysis results using proper SQLModel deletion pattern.

        Deletes all FOVs (which cascades to ROIs, Traces, and DataAnalysis).
        Wells are preserved since they contain plate map metadata and conditions.

        This uses explicit session.delete() calls as recommended by SQLModel docs.
        Following SQLModel tutorial: https://sqlmodel.tiangolo.com/tutorial/delete/
        """
        if self._engine is None:
            cali_logger.error(
                "No database engine found. Cannot clear analysis results. "
                "Please set an experiment first with set_experiment(Experiment)."
            )
            return

        from sqlmodel import Session, select

        with Session(self._engine) as session:
            # Query all FOVs for this experiment using joins
            # Deleting FOVs will cascade to ROIs, Traces, and DataAnalysis
            statement = select(FOV).join(Well).join(Plate)

            fovs = session.exec(statement).all()

            cali_logger.info(
                f"Deleting {len(fovs)} FOVs and their associated analysis data..."
            )

            # Explicitly delete each FOV (cascade will handle ROIs/Traces/DataAnalysis)
            for fov in fovs:
                session.delete(fov)

            session.commit()

        cali_logger.info("💾 Analysis results cleared from database.")

    # PRIVATE METHODS ----------------------------------------------------------------

    def _log_and_emit(self, msg: str, type: str = "info") -> None:
        """Log and display an error message."""
        if type == "info":
            cali_logger.info(msg)
        elif type == "error":
            cali_logger.error(msg)
        elif type == "warning":
            cali_logger.warning(msg)
        elif type == "debug":
            cali_logger.debug(msg)
        self.analysisInfo.emit(msg, type)

    def _on_analysis_finished(self) -> None:
        """Log completion and show any failed labels."""
        cali_logger.info("Analysis Finished.")

        # show a message box if there are failed labels
        if self._failed_labels:
            msg = (
                "The following labels were not found during the analysis:\n\n"
                + "\n".join(self._failed_labels)
            )
            self._log_and_emit(msg, "error")

    def _extract_traces_data(self) -> None:
        """Extract the roi traces in multiple threads."""
        cali_logger.info("Starting traces analysis...")

        exp = self.experiment()
        if exp is None or exp.analysis_settings is None:
            cali_logger.error("Experiment or AnalysisSettings not set.")
            return

        # Load stimulation mask if available for evoked experiments
        settings = exp.analysis_settings
        if exp.experiment_type == EVOKED:
            self._stimulated_area_mask = settings.stimulated_mask_area()
        else:
            # Ensure attribute exists even if no stimulation mask
            self._stimulated_area_mask = None

        # set number of threads to use
        threads = settings.threads
        cali_logger.info(f"Number of threads for analysis: {threads}")

        positions = exp.positions_analyzed
        cali_logger.info(f"Positions to analyze: {positions}")
        try:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                # Check for cancellation before submitting futures
                if self._cancellation_event.is_set():
                    cali_logger.info(
                        "Cancellation requested before starting thread pool"
                    )
                    return

                futures = [
                    executor.submit(self._analyze_position, p) for p in positions
                ]

                for future in as_completed(futures):
                    # Check for cancellation at the start of each iteration
                    if self._cancellation_event.is_set():
                        cali_logger.info(
                            "Cancellation requested, shutting down executor..."
                        )
                        # Cancel pending futures and shutdown executor
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    try:
                        future.result()

                        # Check for cancellation after each completed position
                        if self._cancellation_event.is_set():
                            cali_logger.info("Cancellation requested after position")
                            break

                    except Exception as e:
                        self._log_and_emit(f"An error occurred: {e}", "error")
                        break

            # Check if cancelled before finishing
            if self._cancellation_event.is_set():
                cali_logger.info("Run Cancelled")
                return

            self._on_analysis_finished()

        except Exception as e:
            cali_logger.error(f"An error occurred: {e}")
            self._log_and_emit(f"An error occurred: {e}", "error")

    def _analyze_position(self, p: int) -> None:
        """Extract the roi traces for the given position.

        This method works with database session to ensure all changes are persisted.
        """
        if self._engine is None or self._data is None:
            return
        if self._check_for_abort_requested():
            return

        from sqlmodel import Session, select

        with Session(self._engine) as session:
            # Get experiment from database within session
            exp = session.exec(select(Experiment)).first()
            if exp is None:
                return

            self._extract_trace_data_per_position(session, exp, p)

    # -------------------------------------------------------------------

    def _extract_trace_data_per_position(
        self, session: Session, exp: Experiment, p: int
    ) -> None:
        """Implementation of trace extraction with session context."""
        if self._data is None or self._check_for_abort_requested():
            return

        # get the data and metadata for the position
        data, meta = self._data.isel(p=p, metadata=True)

        # get the fov_name name from metadata
        fov_name = self._get_fov_name(EVENT_KEY, meta, p)

        # get the labels file for the position
        labels_path = self._get_labels_file_for_position(fov_name, p)
        if labels_path is None:
            return

        # open the labels file and create masks for each label
        labels = tifffile.imread(labels_path)

        # Check for cancellation after file I/O operation
        if self._check_for_abort_requested():
            return

        # { roi_id -> np.ndarray mask }
        labels_masks = self._create_label_masks_dict(labels)
        sequence = cast("useq.MDASequence", self._data.sequence)

        # Check for cancellation after loading and processing labels
        if self._check_for_abort_requested():
            return

        settings = exp.analysis_settings
        assert settings is not None  # this should never be None here

        # Prepare masks for neuropil correction if enabled
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
        elapsed_time_list = self.get_elapsed_time_list(meta)
        # if the elapsed time is not available or for any reason is different from
        # the number of timepoints, set it as list of timepoints every exp_time
        if len(elapsed_time_list) != timepoints:
            elapsed_time_list = [i * exp_time for i in range(timepoints)]
        # get the total time in seconds for the recording
        tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0]) / 1000

        # check if it is an evoked activity experiment
        evoked_experiment = exp.experiment_type == EVOKED

        # >>>> HERE is the big loop over roi mask, calling _process_roi_trace
        msg = f"Extracting Traces Data from Well {fov_name}."
        cali_logger.info(msg)
        for label_value, _label_mask in tqdm(labels_masks.items(), desc=msg):
            if self._check_for_abort_requested():
                cali_logger.info(
                    f"Cancellation requested during processing of {fov_name}"
                )
                break

            # extract the data - pass session and exp
            self._process_roi_trace(
                session,
                exp,
                data,
                meta,
                fov_name,
                p,  # Pass the actual position index
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

        # Commit all changes after processing all ROIs for this position
        if not self._check_for_abort_requested():
            session.commit()
            cali_logger.info(f"💾 Updated analysis data for position {p} in database.")

    def _ensure_list_loaded(self, obj: object, attr_name: str) -> list:
        """Safely ensure a relationship list is loaded and initialized.

        Parameters
        ----------
        obj : object
            The SQLModel object containing the relationship
        attr_name : str
            The name of the relationship attribute

        Returns
        -------
        list
            The loaded list (may be empty if lazy load failed)
        """
        try:
            attr_list = getattr(obj, attr_name)
            if attr_list is None:
                setattr(obj, attr_name, [])
                attr_list = getattr(obj, attr_name)
        except Exception:
            # Lazy load failed, initialize empty list
            setattr(obj, attr_name, [])
            attr_list = getattr(obj, attr_name)
        return attr_list

    def _get_or_create_well(
        self, session: Session, exp: Experiment, well_name: str
    ) -> Well:
        """Get existing well or create new one.

        Parameters
        ----------
        session : Session
            Active database session
        exp : Experiment
            Experiment object (attached to session)
        well_name : str
            Well name (e.g., "B5")

        Returns
        -------
        Well
            Existing or newly created Well object
        """
        if exp.plate is None:
            raise ValueError("Experiment plate not initialized")

        wells_list = self._ensure_list_loaded(exp.plate, "wells")

        # Find existing well
        for well in wells_list:
            if well.name == well_name:
                return well

        # Create new well
        row = ord(well_name[0]) - ord("A")
        col = int(well_name[1:]) - 1
        well = Well(
            plate_id=0,  # Placeholder, will be set via relationship
            name=well_name,
            row=row,
            column=col,
            fovs=[],
        )
        well.plate = exp.plate
        wells_list.append(well)
        session.add(well)  # Explicitly add to session
        return well

    def _get_or_create_fov(
        self, session: Session, well: Well, fov_name: str, pos_idx: int
    ) -> FOV:
        """Get existing FOV or create new one.

        Parameters
        ----------
        session : Session
            Active database session
        well : Well
            Parent well
        fov_name : str
            FOV name (e.g., "B5_0000_p0")
        pos_idx : int
            Position index

        Returns
        -------
        FOV
            Existing or newly created FOV object
        """
        fovs_list = self._ensure_list_loaded(well, "fovs")

        # Find existing FOV
        for fov in fovs_list:
            if fov.name == fov_name:
                return fov

        # Create new FOV
        fov = FOV(
            name=fov_name,
            position_index=pos_idx,
            fov_number=pos_idx,
            rois=[],
        )
        fov.well = well
        session.add(fov)  # Explicitly add to session
        return fov

    def _get_or_create_roi(
        self,
        session: Session,
        fov: FOV,
        label_value: int,
        settings: AnalysisSettings,
        is_active: bool,
        is_stimulated: bool,
        mask_coords: tuple[list[int], list[int]],
        mask_shape: tuple[int, int],
        neuropil_mask_coords: tuple[list[int], list[int]] | None,
        neuropil_mask_shape: tuple[int, int] | None,
    ) -> ROI:
        """Get existing ROI or create new one with masks.

        Parameters
        ----------
        session : Session
            Active database session
        fov : FOV
            Parent FOV
        label_value : int
            ROI label value
        settings : AnalysisSettings
            Analysis settings to associate with ROI
        is_active : bool
            Whether ROI has detected peaks
        is_stimulated : bool
            Whether ROI overlaps with stimulation area
        mask_coords : tuple[list[int], list[int]]
            ROI mask coordinates (y, x)
        mask_shape : tuple[int, int]
            ROI mask shape (height, width)
        neuropil_mask_coords : tuple[list[int], list[int]] | None
            Neuropil mask coordinates (y, x) or None
        neuropil_mask_shape : tuple[int, int] | None
            Neuropil mask shape (height, width) or None

        Returns
        -------
        ROI
            Existing or newly created ROI object
        """
        rois_list = self._ensure_list_loaded(fov, "rois")

        # Find existing ROI
        for roi in rois_list:
            if roi.label_value == label_value:
                # Update existing ROI
                roi.sqlmodel_update(
                    {
                        "active": is_active,
                        "stimulated": is_stimulated,
                        "analysis_settings": settings,
                    }
                )

                # Update masks
                if roi.roi_mask:
                    roi.roi_mask.sqlmodel_update(
                        {
                            "coords_y": mask_coords[0],
                            "coords_x": mask_coords[1],
                            "height": mask_shape[0],
                            "width": mask_shape[1],
                        }
                    )
                else:
                    roi.roi_mask = Mask(
                        coords_y=mask_coords[0],
                        coords_x=mask_coords[1],
                        height=mask_shape[0],
                        width=mask_shape[1],
                        mask_type="roi",
                    )

                if neuropil_mask_coords and neuropil_mask_shape:
                    if roi.neuropil_mask:
                        roi.neuropil_mask.sqlmodel_update(
                            {
                                "coords_y": neuropil_mask_coords[0],
                                "coords_x": neuropil_mask_coords[1],
                                "height": neuropil_mask_shape[0],
                                "width": neuropil_mask_shape[1],
                            }
                        )
                    else:
                        roi.neuropil_mask = Mask(
                            coords_y=neuropil_mask_coords[0],
                            coords_x=neuropil_mask_coords[1],
                            height=neuropil_mask_shape[0],
                            width=neuropil_mask_shape[1],
                            mask_type="neuropil",
                        )

                return roi

        # Create new ROI with masks
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

        roi = ROI(
            fov_id=0,  # Placeholder, will be set via relationship
            label_value=label_value,
            active=is_active,
            stimulated=is_stimulated,
            roi_mask=roi_mask,
            neuropil_mask=neuropil_mask_obj,
        )
        roi.fov = fov
        roi.analysis_settings = settings
        session.add(roi)  # Explicitly add to session
        return roi

    def _process_roi_trace(
        self,
        session: Session,
        exp: Experiment,
        data: np.ndarray,
        meta: list[dict],
        fov_name: str,
        pos_idx: int,
        label_value: int,
        label_mask: np.ndarray,
        tot_time_sec: float,
        evoked_exp: bool,
        elapsed_time_list: list[float],
        neuropil_mask: np.ndarray | None = None,
        neuropil_correction_factor: float | None = None,
    ) -> None:
        """Process individual ROI traces and update the experiment.

        Works with session-attached objects to ensure proper persistence.
        """
        # Early exit if cancellation is requested
        if self._check_for_abort_requested():
            return

        # exp is already passed in and attached to session
        if exp.analysis_settings is None:
            return

        settings = exp.analysis_settings

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
            return

        # check if the roi is stimulated
        roi_stimulation_overlap_ratio = 0.0
        if evoked_exp and self._stimulated_area_mask is not None:
            roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
                self._stimulated_area_mask, label_mask
            )

        # compute the mean for each frame
        roi_trace_uncorrected: np.ndarray = masked_data.mean(axis=1)
        win = settings.dff_window

        # Check for cancellation before DFF calculation
        if self._check_for_abort_requested():
            return

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
        if self._check_for_abort_requested():
            return

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
        # cali_logger.info(
        #     f"Decay constant: {_t} seconds, "
        #     f"Sampling frequency: {len(roi_trace) / tot_time_sec} Hz"
        # )

        # Check for cancellation after deconvolution
        if self._check_for_abort_requested():
            return

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
            return

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
            return

        # find peaks in the deconvolved trace
        peaks_dec_dff, _ = find_peaks(
            dec_dff,
            prominence=peaks_prominence_dec_dff,
            height=peaks_height_dec_dff,
            distance=min_distance_frames,
        )
        peaks_dec_dff = cast("np.ndarray", peaks_dec_dff)

        # Check for cancellation after peak finding
        if self._check_for_abort_requested():
            return

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
            return

        # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
        iei = get_iei(peaks_dec_dff, elapsed_time_list)

        # get mask coords and shape for the ROI
        mask_coords, mask_shape = mask_to_coordinates(label_mask)

        # get neuropil mask coords and shape if neuropil mask exists
        neuropil_mask_coords = neuropil_mask_shape = None
        if neuropil_mask is not None:
            neuropil_mask_coords, neuropil_mask_shape = mask_to_coordinates(
                neuropil_mask
            )

        # Build SQLModel objects to update the experiment
        # Get or create Well, FOV for this ROI
        well_name = fov_name.split("_")[0]
        # pos_idx is now passed as a parameter, no need to recalculate

        # Use helper methods to get or create hierarchy objects
        well = self._get_or_create_well(session, exp, well_name)
        fov = self._get_or_create_fov(session, well, fov_name, pos_idx)
        roi = self._get_or_create_roi(
            session,
            fov,
            label_value,
            settings,
            is_active=len(peaks_dec_dff) > 0,
            is_stimulated=is_roi_stimulated,
            mask_coords=mask_coords,
            mask_shape=mask_shape,
            neuropil_mask_coords=neuropil_mask_coords,
            neuropil_mask_shape=neuropil_mask_shape,
        )

        # Create or update Traces
        if roi.traces is None:
            traces = Traces(
                roi=roi,
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
            roi.traces = traces
            session.add(traces)  # Explicitly add to session
        else:
            # Update existing traces
            roi.traces.sqlmodel_update(
                {
                    "raw_trace": cast("list[float]", roi_trace_uncorrected.tolist()),
                    "corrected_trace": cast("list[float]", roi_trace.tolist()),
                    "neuropil_trace": (
                        cast("list[float]", neuropil_trace.tolist())
                        if neuropil_trace is not None
                        else None
                    ),
                    "dff": cast("list[float]", dff.tolist()),
                    "dec_dff": dec_dff.tolist(),
                    "x_axis": elapsed_time_list,
                }
            )

        # Create or update DataAnalysis with ROI-specific thresholds
        if roi.data_analysis is None:
            data_analysis = DataAnalysis(
                roi=roi,
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
            roi.data_analysis = data_analysis
            session.add(data_analysis)  # Explicitly add to session
        else:
            # Update existing data analysis
            roi.data_analysis.sqlmodel_update(
                {
                    "cell_size": roi_size,
                    "cell_size_units": "µm" if px_size is not None else "pixel",
                    "total_recording_time_sec": tot_time_sec,
                    "dec_dff_frequency": frequency,
                    "peaks_dec_dff": peaks_dec_dff.tolist(),
                    "peaks_amplitudes_dec_dff": peaks_amplitudes_dec_dff,
                    "iei": iei,
                    "inferred_spikes": spikes.tolist(),
                    "peaks_prominence_dec_dff": peaks_prominence_dec_dff,
                    "peaks_height_dec_dff": peaks_height_dec_dff,
                    "inferred_spikes_threshold": spike_detection_threshold,
                }
            )

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_event.is_set()

    def _get_fov_name(self, event_key: str, meta: list[dict], p: int) -> str:
        """Retrieve the fov name from metadata."""
        # the "Event" key was used in the old metadata format
        pos_name = meta[0].get(event_key, {}).get("pos_name", f"pos_{str(p).zfill(4)}")
        return f"{pos_name}_p{p}"

    def _get_labels_file_for_position(self, fov: str, p: int) -> str | None:
        """Retrieve the labels file for the given position."""
        # if the fov name does not end with "_p{p}", add it
        labels_name = f"{fov}.tif" if fov.endswith(f"_p{p}") else f"{fov}_p{p}.tif"
        labels_path = self._get_labels_file(labels_name)
        if labels_path is None:
            self._failed_labels.append(labels_name)
            cali_logger.error("No labels found for %s!", labels_name)
        return labels_path

    def _get_labels_file(self, label_name: str) -> str | None:
        """Get the labels file for the given name."""
        exp = self.experiment()
        if exp is None:
            return None
        if (labels_path := exp.labels_path) is None:
            return None
        for label_file in Path(labels_path).glob("*.tif"):
            if label_file.name.endswith(label_name):
                return str(label_file)
        return None

    def _create_label_masks_dict(self, labels: np.ndarray) -> dict[int, np.ndarray]:
        """Create masks for each label in the labels image."""
        # get the range of labels and remove the background (0)
        labels_range = np.unique(labels[labels != 0])
        # Convert numpy int to Python int to avoid bytes serialization issues
        return {
            int(label_value): (labels == label_value) for label_value in labels_range
        }

    def get_elapsed_time_list(self, meta: list[dict]) -> list[float]:
        elapsed_time_list: list[float] = []
        # get the elapsed time for each timepoint to calculate tot_time_sec
        if RUNNER_TIME_KEY in meta[0]:  # new metadata format
            for m in meta:
                rt = m[RUNNER_TIME_KEY]
                if rt is not None:
                    elapsed_time_list.append(float(rt))
        return elapsed_time_list
