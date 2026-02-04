"""Unified runner interface for detection and analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import event
from sqlmodel import Session, create_engine, select

from cali._constants import DEFAULT_CALI_DB_NAME, CorrelationDataType, TraceDataType
from cali.detection import DetectionRunner
from cali.extraction import ExtractionRunner
from cali.logger import cali_logger
from cali.readers._tiff_collection_reader import TiffCollectionReader
from cali.sqlmodel import save_experiment_to_database
from cali.sqlmodel._model import FOV, ROI, CaliResult, Traces
from cali.util import commit_fov_result, load_data_from_path

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    from cali.readers._ome_zarr_reader import OMEZarrReader
    from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader
    from cali.sqlmodel import (
        AnalysisSettings,
        DetectionSettings,
        Experiment,
        ExtractionSettings,
    )


class CaliRunner:
    """Unified runner for calcium imaging detection and analysis.

    This class provides a single interface for running both detection (ROI
    segmentation) and analysis (trace extraction) steps. It delegates to
    specialized runners internally while providing a cleaner API.

    The runner automatically determines whether detection needs to be run by
    checking if FOVs already exist with the specified detection settings.

    Examples
    --------
    Run detection only:
    >>> runner = CaliRunner()
    >>> runner.run(
    ...     experiment=exp,
    ...     detection_settings=d_settings,
    ...     global_position_indices=[0, 1, 2],
    ... )

    Run detection + analysis:
    >>> runner.run(
    ...     experiment=exp,
    ...     detection_settings=d_settings,
    ...     analysis_settings=a_settings,
    ...     global_position_indices=[0, 1, 2],
    ... )

    Run analysis only (detection already exists):
    >>> # If FOVs already exist, detection is automatically skipped
    >>> runner.run(
    ...     experiment=exp,
    ...     detection_settings=d_settings,  # Specify which detection to use
    ...     analysis_settings=a_settings,
    ...     global_position_indices=[0, 1, 2],
    ... )

    Force re-run detection:
    >>> # Use force=True to delete existing results and re-run
    >>> runner.run(
    ...     experiment=exp,
    ...     detection_settings=d_settings,
    ...     global_position_indices=[0, 1, 2],
    ...     force=True,
    ... )

    Adjust commit frequency:
    >>> # Commit every 5 FOVs instead of default 10
    >>> runner = CaliRunner(commit_batch_size=5)
    """

    def __init__(self, commit_batch_size: int = 10) -> None:
        """Initialize the unified runner.

        Parameters
        ----------
        commit_batch_size : int
            Number of FOVs to accumulate before committing to database.
            Default is 10. Set to 1 for immediate commits (safest but slowest).
        """
        self._db_path: Path | None = None
        self.commit_batch_size = commit_batch_size

        # Internal runners
        self._detection_runner = DetectionRunner()
        self._extraction_runner = ExtractionRunner()

    @property
    def database_path(self) -> str | None:
        """Get the database path."""
        return str(self._db_path) if self._db_path is not None else None

    def cancel(self) -> None:
        """Cancel both detection and extraction processes."""
        self._detection_runner.cancel()
        self._extraction_runner.cancel()

    def run(
        self,
        experiment: Experiment,
        dataset_path: str | Path,
        detection_settings: DetectionSettings | int,
        *,
        extraction_settings: ExtractionSettings | int | None = None,
        analysis_settings: AnalysisSettings | int | None = None,
        global_position_indices: Sequence[int] | None = None,
        database_name: str | None = None,
        output_path: str | Path | None = None,
        overwrite: bool = False,
        force: bool = False,
        echo: bool = False,
        as_generator: bool = False,
        export_traces: dict[TraceDataType, bool] | None = None,
        export_correlations: dict[CorrelationDataType, bool] | None = None,
    ) -> Generator[str, None, None] | None:
        """Run detection and/or analysis on the experiment.

        This method orchestrates the entire pipeline:
        0. Makes sure data are ready
        1. Sets up the database (creates if needed)
        2. Deduplicates settings (reuses existing via pointers)
        3. Automatically determines if detection needed (checks if FOVs exist)
        4. Runs detection only if needed (or if force=True)
        5. Commits detection results to database
        6. Runs analysis (if settings provided)
        7. Commits analysis results to database
        8. Creates CaliResult entry (always new, never overwrites)

        Parameters
        ----------
        experiment : Experiment
            Experiment to process
        dataset_path: str | Path
            Data path to the raw imaging data (zarr/tensorstore).
        detection_settings : DetectionSettings | int
            Detection parameters (required to specify which ROIs to analyze). It can be
            either a DetectionSettings instance or an integer ID referencing
            an existing DetectionSettings in the database.
        extraction_settings : ExtractionSettings | int | None
            By default None.
            Extraction parameters (neuropil, dff_window, decay_constant, etc.).
            Required if analysis_settings is provided. Can be an ExtractionSettings
            instance or integer ID referencing existing settings in the database.
        analysis_settings : AnalysisSettings | int | None
            By default None.
            Analysis parameters (peak detection, thresholds, etc.).
            If None, only extraction (traces) is performed. If provided,
            both extraction and analysis are run.
        global_position_indices : Sequence[int] | None
            Position indices to process. If None, processes all positions
            in the dataset.
        database_name : str | None
            Name of the database file to create/use. If None, defaults "results.cali".
        output_path : Path | None
            Output path to save database and analysis results. If None, uses dataset
            parent directory.
        overwrite : bool
            Whether to overwrite existing database
        force : bool
            Whether to force re-run even if results exist.
            If True, existing results for the specified settings and positions
            will be deleted and re-computed.
        echo : bool
            Enable SQLAlchemy echo for database operations
        as_generator : bool
            If True, returns a Generator that yields progress strings.
            If False (default), executes directly and silently consumes
            the generator.
        export_traces : dict[TraceDataType, bool] | None
            Optional dictionary mapping trace type names to export flags.
            Keys must be valid TraceDataType literals (e.g., "Raw Calcium Traces).
            If provided, exports selected traces to CSV after extraction completes.
            Example: {"Raw Calcium Traces": True, "ΔF/F Traces": True}
        export_correlations : dict[CorrelationDataType, bool] | None
            Optional dictionary mapping correlation data type names to export flags.
            Keys must be valid CorrelationDataType literals (e.g.
            "ΔF/F Correlation Matrix"). If provided, exports selected correlation data
            to CSV after analysis completes. Example: {"ΔF/F Correlation Matrix": True}

        Returns
        -------
        Generator[str, None, None] | None
            If as_generator=True, returns a generator yielding progress strings.

        Raises
        ------
        ValueError
            If no FOVs exist for the specified detection settings and positions
        """
        generator = self._run_generator(
            experiment=experiment,
            dataset_path=dataset_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings,
            global_position_indices=global_position_indices,
            database_name=database_name,
            output_path=output_path,
            overwrite=overwrite,
            force=force,
            echo=echo,
            export_traces=export_traces,
            export_correlations=export_correlations,
        )

        # Return generator directly if requested
        if as_generator:
            return generator

        # Otherwise, consume generator silently
        for _ in generator:
            pass
        return None

    def _run_generator(
        self,
        experiment: Experiment,
        dataset_path: str | Path,
        detection_settings: DetectionSettings | int,
        *,
        extraction_settings: ExtractionSettings | int | None = None,
        analysis_settings: AnalysisSettings | int | None = None,
        global_position_indices: Sequence[int] | None = None,
        database_name: str | None = None,
        output_path: str | Path | None = None,
        overwrite: bool = False,
        force: bool = False,
        echo: bool = False,
        export_traces: dict[TraceDataType, bool] | None = None,
        export_correlations: dict[CorrelationDataType, bool] | None = None,
    ) -> Generator[str, None, None]:
        """Internal generator for run progress.

        Yields progress strings during execution.
        """
        # Reset cancellation events at the start of each run
        self._detection_runner._cancellation_event.clear()
        self._extraction_runner._cancellation_event.clear()

        # 0. Make sure data are ready
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader | None
        tiff_settings = experiment.tiff_collection_settings(dataset_path)
        if tiff_settings is not None:
            dataset = TiffCollectionReader(tiff_settings)
        else:
            dataset = load_data_from_path(dataset_path)

        if dataset is None:
            msg = f"❌ Could not load data from path: {dataset_path}"
            cali_logger.error(msg)
            raise ValueError(msg)
        if dataset.sequence is None:
            msg = "❌  Dataset does not contain sequence information."
            cali_logger.error(msg)
            raise ValueError(msg)

        if output_path is None:
            output_path = Path(dataset_path).parent
        elif isinstance(output_path, str):
            output_path = Path(output_path)

        # 1. Setup database
        if database_name is not None:
            db_name = (
                database_name
                if database_name.endswith(".cali")
                else f"{database_name}.cali"
            )
            self._db_path = output_path / db_name
        else:
            self._db_path = output_path / DEFAULT_CALI_DB_NAME
        self._setup_database(self._db_path, experiment, overwrite)

        # 2. Get database engine and session
        engine = create_engine(
            f"sqlite:///{self._db_path}",
            echo=echo,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )

        # Enable foreign keys for SQLite
        @event.listens_for(engine, "connect")  # type: ignore
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        try:
            with Session(engine) as session:
                # 3. Deduplicate and persist settings
                detection_settings = self._get_or_create_detection_settings(
                    session, detection_settings
                )

                # Validate extraction/analysis settings combination
                if analysis_settings is not None and extraction_settings is None:
                    raise ValueError(
                        "extraction_settings is required when "
                        "analysis_settings is provided"
                    )

                # Narrow types to actual settings objects
                extraction_settings_obj: ExtractionSettings | None = None
                analysis_settings_obj: AnalysisSettings | None = None

                if extraction_settings is not None:
                    extraction_settings_obj = self._get_or_create_extraction_settings(
                        session, extraction_settings
                    )
                    extraction_settings_id = extraction_settings_obj.id
                    extraction_threads = extraction_settings_obj.threads
                    assert extraction_settings_id is not None
                else:
                    extraction_settings_id = None
                    extraction_threads = None

                if analysis_settings is not None:
                    analysis_settings_obj = self._get_or_create_analysis_settings(
                        session, analysis_settings
                    )
                    analysis_settings_id = analysis_settings_obj.id
                    assert analysis_settings_id is not None
                else:
                    analysis_settings_id = None

                det_id = detection_settings.id
                assert det_id is not None

                # 4. Determine which positions need detection
                # Track whether user explicitly provided position indices
                user_provided_positions = global_position_indices is not None

                if global_position_indices is None:
                    if dataset.sequence is None:
                        msg = "Dataset sequence metadata is missing."
                        cali_logger.error(msg)
                        raise ValueError(msg)
                    # Process all positions
                    global_position_indices = list(
                        range(len(dataset.sequence.stage_positions))
                    )

                positions_for_detection = self._get_positions_for_detection(
                    session, det_id, global_position_indices, force=force
                )

                if force and positions_for_detection:
                    self._delete_detection_results(
                        session, det_id, list(positions_for_detection)
                    )

                yield "🔍 Running Detection..."

                # 5. Run detection if needed
                # Initialize result IDs at the beginning to avoid UnboundLocalError
                analysis_result_id: int | None = None
                detection_result_id: int | None = None
                positions_processed_detection = []
                total_rois_detected = 0

                # Create detection-only result if no extraction/analysis will follow
                needs_detection_result = (
                    positions_for_detection
                    and extraction_settings is None
                    and analysis_settings is None
                    and experiment.id is not None
                )
                detection_result_was_created = False
                if needs_detection_result:
                    # Optimistic: assume all will complete
                    detection_result_id, detection_result_was_created = (
                        self._create_or_update_analysis_result(
                            session=session,
                            experiment_id=experiment.id,  # type: ignore
                            detection_settings_id=det_id,
                            extraction_settings_id=None,
                            analysis_settings_id=None,
                            positions_detected=list(positions_for_detection),
                        )
                    )
                if positions_for_detection:
                    yield f"PROGRESS:RESET:{len(positions_for_detection)}"
                    fov_count = 0
                    for fov in self._run_detection(
                        dataset,
                        detection_settings,
                        positions_for_detection,
                    ):
                        # Count ROIs before commit (they may become detached after)
                        roi_count = len(fov.rois)
                        total_rois_detected += roi_count
                        fov_count += 1
                        yield "PROGRESS:UPDATE"

                        # Capture position index before expunging
                        pos_idx = fov.position_index

                        # Commit in batches
                        should_commit = fov_count % self.commit_batch_size == 0
                        if should_commit:
                            cali_logger.info(
                                f"💾 Committing batch of {self.commit_batch_size} FOVs "
                                f"(total: {fov_count}/{len(positions_for_detection)})"
                                "..."
                            )
                        commit_fov_result(
                            session,
                            experiment,
                            fov,
                            det_id,
                            commit=should_commit,
                        )
                        if should_commit:
                            cali_logger.info(
                                f"💾 Committed batch of {self.commit_batch_size} FOVs "
                                f"(total: {fov_count}/{len(positions_for_detection)})"
                            )
                            # Expunge FOV only after commit to free memory
                            # Must check if fov is still in session first
                            if fov in session:
                                session.expunge(fov)

                        positions_processed_detection.append(pos_idx)

                    # Final commit for any remaining FOVs
                    if fov_count % self.commit_batch_size != 0:
                        remaining = fov_count % self.commit_batch_size
                        cali_logger.info(
                            f"💾 Committing final batch of {remaining} FOVs..."
                        )
                        session.commit()
                        cali_logger.info(
                            f"💾 Committed final batch of {remaining} FOVs."
                        )
                        # No need to expunge settings - they stay attached throughout

                    # Log detection completion
                    if positions_processed_detection:
                        cali_logger.info(
                            f"✅ Detection committed: {total_rois_detected} "
                            f"ROIs across {len(positions_processed_detection)} FOVs"
                        )

                    # Update detection-only result with actually completed positions
                    if detection_result_id is not None:
                        result = session.get(CaliResult, detection_result_id)
                        if result:
                            if detection_result_was_created:
                                # Replace: result was created in this run
                                completed = sorted(positions_processed_detection)
                                result.positions_detected = completed
                            else:
                                # Merge: result existed from previous run
                                old_positions = set(result.positions_detected or [])
                                new_positions = set(positions_processed_detection)
                                completed = sorted(old_positions | new_positions)
                                result.positions_detected = completed
                            session.add(result)
                            session.commit()
                            cali_logger.info(
                                f"📝 Updated CaliResult ID {detection_result_id} "
                                f"with completed detected positions: {completed}"
                            )

                # Check for cancellation after detection completes
                # If cancelled and we were planning to run extraction/analysis,
                # we need to create a detection-only result for the completed positions
                if self._detection_runner._cancellation_event.is_set():
                    if (
                        positions_processed_detection
                        and detection_result_id is None
                        and experiment.id is not None
                    ):
                        # Create a detection-only result for what we completed
                        detection_result_id, _ = self._create_or_update_analysis_result(
                            session=session,
                            experiment_id=experiment.id,
                            detection_settings_id=det_id,
                            extraction_settings_id=None,
                            analysis_settings_id=None,
                            positions_detected=sorted(positions_processed_detection),
                        )
                        session.commit()
                        cali_logger.info(
                            f"📝 Created CaliResult ID {detection_result_id} for "
                            f"cancelled run with detected positions: "
                            f"{sorted(positions_processed_detection)}"
                        )
                    return

                # 7. Run extraction if settings provided
                if extraction_settings_obj is not None:
                    # Eager load scalars before detaching for thread safety
                    _ = (
                        extraction_settings_obj.threads,
                        extraction_settings_obj.neuropil_inner_radius,
                        extraction_settings_obj.neuropil_min_pixels,
                        extraction_settings_obj.neuropil_correction_factor,
                        extraction_settings_obj.decay_constant,
                        extraction_settings_obj.dff_window,
                    )
                    session.expunge(extraction_settings_obj)

                    if analysis_settings_obj is not None:
                        # Eager load scalars before detaching
                        _ = (
                            analysis_settings_obj.threads,
                            analysis_settings_obj.peaks_height_value,
                            analysis_settings_obj.peaks_height_mode,
                            analysis_settings_obj.peaks_distance,
                            analysis_settings_obj.peaks_prominence_multiplier,
                            analysis_settings_obj.spike_threshold_value,
                            analysis_settings_obj.spike_threshold_mode,
                            analysis_settings_obj.experiment_type,
                            analysis_settings_obj.stimulation_mask_path,
                        )

                        # Load stimulation mask from file if path provided but
                        # mask ID not yet set
                        if (
                            analysis_settings_obj.stimulation_mask_path
                            and analysis_settings_obj.stimulation_mask_id is None
                        ):
                            import tifffile

                            from cali.sqlmodel._model import Mask
                            from cali.util import mask_to_coordinates

                            stim_mask_file = Path(
                                analysis_settings_obj.stimulation_mask_path
                            )
                            if stim_mask_file.exists():
                                # Load and convert stimulation mask
                                stim_mask_array = tifffile.imread(str(stim_mask_file))
                                coords, shape = mask_to_coordinates(
                                    stim_mask_array.astype(bool)
                                )

                                # Create and attach Mask object
                                stimulation_mask = Mask(
                                    coords_y=coords[0],
                                    coords_x=coords[1],
                                    height=shape[0],
                                    width=shape[1],
                                    mask_type="stimulation",
                                )
                                session.add(stimulation_mask)
                                session.flush()  # Get the mask ID
                                analysis_settings_obj.stimulation_mask = (
                                    stimulation_mask
                                )
                                analysis_settings_obj.stimulation_mask_id = (
                                    stimulation_mask.id
                                )
                                session.add(analysis_settings_obj)
                                session.commit()
                                cali_logger.info(
                                    f"🎭 Loaded stimulation mask from "
                                    f"{stim_mask_file.name}"
                                )

                        # Eagerly load stimulation_mask relationship before detaching
                        # This prevents lazy loading errors in threads
                        if analysis_settings_obj.stimulation_mask is not None:
                            # Access all mask attributes to force SQLAlchemy to
                            # load them
                            _ = analysis_settings_obj.stimulation_mask.coords_y
                            _ = analysis_settings_obj.stimulation_mask.coords_x
                            _ = analysis_settings_obj.stimulation_mask.height
                            _ = analysis_settings_obj.stimulation_mask.width

                    if analysis_settings_obj is not None:
                        # Detach for thread safety
                        session.expunge(analysis_settings_obj)

                    yield "📈 Running Extraction" + (
                        " and 📊 Analysis..." if analysis_settings_obj else "..."
                    )

                    # Determine which positions need extraction/analysis
                    positions_need_extraction = set(
                        self._get_positions_for_extraction(
                            session,
                            det_id,
                            extraction_settings_id,  # type: ignore
                            global_position_indices,
                            force=force,
                        )
                    )

                    # Check which positions need analysis (if analysis is requested)
                    positions_need_analysis = set()
                    if analysis_settings_id is not None:
                        positions_need_analysis = set(
                            self._get_positions_for_analysis(
                                session,
                                det_id,
                                extraction_settings_id,  # type: ignore
                                analysis_settings_id,
                                global_position_indices,
                                force=force,
                            )
                        )

                    # Combine all positions that need processing
                    positions_to_process = sorted(
                        positions_need_extraction | positions_need_analysis
                    )

                    if not positions_to_process:
                        return

                    # Log information about position distribution
                    positions_only_analysis = (
                        positions_need_analysis - positions_need_extraction
                    )
                    if positions_only_analysis:
                        positions_list = sorted(positions_only_analysis)
                        if positions_only_analysis == positions_need_analysis:
                            # ALL positions are analysis-only
                            cali_logger.info(
                                f"⚠️ Extraction already exists for all positions "
                                f"with DetectionSettings ID {det_id} and "
                                f"ExtractionSettings ID {extraction_settings_id}. "
                                "Running analysis only on these positions."
                            )
                        else:
                            # SOME positions are analysis-only
                            cali_logger.info(
                                f"⚠️ Extraction already exists for positions "
                                f"{positions_list}. "
                                "Running analysis only on these positions."
                            )

                    yield f"PROGRESS:RESET:{len(positions_to_process)}"

                    # Create CaliResult FIRST with expected positions so we have the ID
                    analysis_result_was_created = False
                    if experiment.id is not None:
                        # For extraction/analysis, we track both stages
                        if analysis_settings_id is not None:
                            # Full pipeline: track detected, extracted and analyzed
                            analysis_result_id, analysis_result_was_created = (
                                self._create_or_update_analysis_result(
                                    session=session,
                                    experiment_id=experiment.id,
                                    detection_settings_id=det_id,
                                    extraction_settings_id=extraction_settings_id,
                                    analysis_settings_id=analysis_settings_id,
                                    positions_detected=list(positions_to_process),
                                    positions_extracted=list(positions_to_process),
                                    positions_analyzed=list(positions_to_process),
                                )
                            )
                        else:
                            # Extraction only: track detected and extracted
                            analysis_result_id, analysis_result_was_created = (
                                self._create_or_update_analysis_result(
                                    session=session,
                                    experiment_id=experiment.id,
                                    detection_settings_id=det_id,
                                    extraction_settings_id=extraction_settings_id,
                                    analysis_settings_id=None,
                                    positions_detected=list(positions_to_process),
                                    positions_extracted=list(positions_to_process),
                                )
                            )

                    if (
                        force
                        and analysis_result_id is not None
                        and positions_to_process
                    ):
                        self._delete_extraction_results(
                            session, analysis_result_id, list(positions_to_process)
                        )

                    # Process in batches
                    # Use a batch size that matches the number of threads to
                    # ensure good utilization without consuming too much memory.
                    # Loading too many FOVs at once can cause OOM errors.
                    assert extraction_threads is not None
                    batch_size = extraction_threads

                    positions_processed = []
                    fov_count = 0

                    for i in range(0, len(positions_to_process), batch_size):
                        # Check for cancellation before each batch
                        if self._extraction_runner._cancellation_event.is_set():
                            cali_logger.info("🛑 Run cancelled during extraction!")
                            return

                        # Track FOVs committed in this batch for final commit logging
                        batch_fov_count = 0
                        batch_positions = positions_to_process[i : i + batch_size]

                        cali_logger.info(
                            f"💿 Loading {len(batch_positions)} FOVs from database..."
                        )
                        batch_fovs = self._load_fovs_from_db(
                            session, det_id, batch_positions
                        )

                        # Detach FOVs from session to allow safe threading
                        # We must do this because SQLAlchemy objects are not thread-safe
                        # and _run_analysis uses a ThreadPoolExecutor.
                        # Since we used selectinload in _load_fovs_from_db, all needed
                        # data (ROIs, masks, traces) is already loaded.
                        for fov in batch_fovs:
                            session.expunge(fov)

                        if not batch_fovs:
                            continue

                        # Split FOVs based on whether they need extraction
                        fovs_need_extraction = [
                            fov
                            for fov in batch_fovs
                            if fov.position_index in positions_need_extraction
                        ]
                        fovs_only_analysis = [
                            fov
                            for fov in batch_fovs
                            if fov.position_index not in positions_need_extraction
                        ]

                        # Log batch processing plan
                        if fovs_need_extraction:
                            n_extraction = len(fovs_need_extraction)
                            cali_logger.info(
                                f"📈 Running extraction on {n_extraction} FOVs..."
                            )
                        if fovs_only_analysis:
                            n_analysis = len(fovs_only_analysis)
                            cali_logger.info(
                                f"📊 Running analysis only on {n_analysis} FOVs "
                                "(extraction already exists)..."
                            )

                        # Run extraction only on FOVs that need it
                        for fov in self._run_extraction(
                            dataset,
                            extraction_settings_obj,
                            analysis_settings_obj if fovs_need_extraction else None,
                            fovs=fovs_need_extraction if fovs_need_extraction else [],
                        ):
                            self._process_fov_results(
                                fov,
                                session,
                                analysis_result_id,
                                include_traces=True,
                            )
                            fov_count += 1
                            batch_fov_count += 1
                            yield "PROGRESS:UPDATE"
                            should_commit = fov_count % self.commit_batch_size == 0
                            if should_commit:
                                cali_logger.info(
                                    f"💾 Committing batch of {self.commit_batch_size} "
                                    f"FOVs (total: {fov_count}/"
                                    f"{len(positions_to_process)})..."
                                )
                            commit_fov_result(
                                session, experiment, fov, commit=should_commit
                            )
                            if should_commit:
                                cali_logger.info(
                                    f"💾 Committed batch of "
                                    f"{self.commit_batch_size} FOVs "
                                    f"(total: {fov_count}/"
                                    f"{len(positions_to_process)})"
                                )
                            positions_processed.append(fov.position_index)

                        # Run analysis-only for FOVs with existing extraction
                        if analysis_settings_obj and fovs_only_analysis:
                            for fov in self._run_analysis_only(
                                analysis_settings_obj,
                                fovs=fovs_only_analysis,
                            ):
                                self._process_fov_results(
                                    fov,
                                    session,
                                    analysis_result_id,
                                    include_traces=False,
                                )
                                fov_count += 1
                                batch_fov_count += 1
                                yield "PROGRESS:UPDATE"
                                should_commit = fov_count % self.commit_batch_size == 0
                                if should_commit:
                                    cali_logger.info(
                                        f"💾 Committing batch of "
                                        f"{self.commit_batch_size} FOVs "
                                        f"(total: {fov_count}/"
                                        f"{len(positions_to_process)})..."
                                    )
                                commit_fov_result(
                                    session, experiment, fov, commit=should_commit
                                )
                                if should_commit:
                                    cali_logger.info(
                                        f"💾 Committed batch of "
                                        f"{self.commit_batch_size} FOVs "
                                        f"(total: {fov_count}/"
                                        f"{len(positions_to_process)})"
                                    )
                                positions_processed.append(fov.position_index)

                        # Commit any remaining in this batch and clear memory
                        # Only commit if there were uncommitted FOVs in this batch
                        uncommitted_count = batch_fov_count % self.commit_batch_size
                        if uncommitted_count > 0:
                            cali_logger.info(
                                f"💾 Committing final batch of {uncommitted_count} "
                                f"FOVs..."
                            )
                            session.commit()
                            cali_logger.info(
                                f"💾 Committed final batch of {uncommitted_count} FOVs."
                            )

                        # Expunge FOVs to free memory
                        for fov in batch_fovs:
                            try:
                                session.expunge(fov)
                            except Exception:
                                pass

                    # Log completion
                    if positions_processed:
                        cali_logger.info(f"✅ Extraction committed: {fov_count} FOVs")

                    # Update analysis result with actually completed positions
                    cali_logger.info(
                        "📝 Updating CaliResult with completed positions..."
                    )
                    if analysis_result_id is not None:
                        result = session.get(CaliResult, analysis_result_id)
                        if result:
                            completed = sorted(positions_processed)
                            if analysis_result_was_created:
                                # Replace: result was created in this run
                                result.positions_extracted = completed
                                if analysis_settings_id is not None:
                                    result.positions_analyzed = completed
                            else:
                                # Merge: result existed from previous run
                                old_extracted = set(result.positions_extracted or [])
                                new_extracted = set(positions_processed)
                                result.positions_extracted = sorted(
                                    old_extracted | new_extracted
                                )
                                if analysis_settings_id is not None:
                                    old_analyzed = set(result.positions_analyzed or [])
                                    result.positions_analyzed = sorted(
                                        old_analyzed | new_extracted
                                    )
                            session.add(result)
                            session.commit()

                            stage = (
                                "extracted and analyzed"
                                if analysis_settings_id is not None
                                else "extracted"
                            )
                            cali_logger.info(
                                f"📝 Updated CaliResult ID {analysis_result_id} "
                                f"with completed {stage} positions: "
                                f"{result.positions_extracted}"
                            )

                    # Export traces to CSV if requested
                    if export_traces and analysis_result_id is not None:
                        from cali.util._database_to_csv import export_traces_to_csv

                        yield "🗂️ Exporting traces to CSV..."
                        export_traces_to_csv(
                            engine,
                            export_traces,
                            analysis_result_id,
                            self._db_path,
                        )
                        yield "🗂️ Exported traces to CSV"
                    # Export correlations to CSV if requested
                    if export_correlations and analysis_result_id is not None:
                        from cali.util._database_to_csv import (
                            export_correlations_to_csv,
                        )

                        yield "🗂️ Exporting correlations to CSV..."
                        export_correlations_to_csv(
                            engine,
                            export_correlations,
                            analysis_result_id,
                            self._db_path,
                            position_indices=(
                                positions_processed if user_provided_positions else None
                            ),
                        )
                        yield "🗂️ Exported correlations to CSV"

        finally:
            cali_logger.info("🏁 Cali Run finished!")
            engine.dispose(close=True)

    # ==================== PRIVATE HELPER METHODS ====================

    def _delete_detection_results(
        self, session: Session, detection_settings_id: int, positions: list[int]
    ) -> None:
        """Delete existing detection results (ROIs) for specific positions."""
        cali_logger.info(
            f"🗑️ Deleting existing detection results for {len(positions)} positions..."
        )

        from cali.sqlmodel import FOV, ROI

        # Select ROIs to delete
        statement = (
            select(ROI)
            .join(FOV)
            .where(
                ROI.detection_settings_id == detection_settings_id,
                FOV.position_index.in_(positions),  # type: ignore
            )
        )
        rois_to_delete = session.exec(statement).all()

        if rois_to_delete:
            for roi in rois_to_delete:
                session.delete(roi)
            session.commit()
            cali_logger.info(f"🗑️ Deleted {len(rois_to_delete)} existing ROIs.")

    def _delete_extraction_results(
        self, session: Session, analysis_result_id: int, positions: list[int]
    ) -> None:
        """Delete existing extraction/analysis results for specific positions."""
        cali_logger.info(
            f"🗑️ Deleting existing extraction/analysis results for "
            f"{len(positions)} positions..."
        )

        from cali.sqlmodel import FOV, ROI, DataAnalysis, Traces

        # Delete Traces
        traces_stmt = (
            select(Traces)
            .join(ROI)
            .join(FOV)
            .where(
                Traces.analysis_result_id == analysis_result_id,
                FOV.position_index.in_(positions),  # type: ignore
            )
        )
        traces_to_delete = session.exec(traces_stmt).all()

        # Delete DataAnalysis
        da_stmt = (
            select(DataAnalysis)
            .join(ROI)
            .join(FOV)
            .where(
                DataAnalysis.analysis_result_id == analysis_result_id,
                FOV.position_index.in_(positions),  # type: ignore
            )
        )
        da_to_delete = session.exec(da_stmt).all()

        count = 0
        if traces_to_delete:
            for t in traces_to_delete:
                session.delete(t)
            count += len(traces_to_delete)

        if da_to_delete:
            for da in da_to_delete:
                session.delete(da)
            count += len(da_to_delete)

        if count > 0:
            session.commit()
            cali_logger.info(
                f"🗑️ Deleted {len(traces_to_delete)} traces and "
                f"{len(da_to_delete)} analysis records."
            )

    def _get_positions_for_detection(
        self,
        session: Session,
        detection_settings_id: int,
        global_position_indices: Sequence[int],
        force: bool = False,
    ) -> list[int]:
        """Get positions that need detection.

        Returns positions that either:
        - Don't have ROIs with this detection settings yet, OR
        - force=True (re-run all)

        Parameters
        ----------
        session : Session
            Database session
        detection_settings_id : int
            Detection settings ID to check
        global_position_indices : Sequence[int]
            Positions to check
        force : bool
            If True, returns all positions regardless of existing results.

        Returns
        -------
        list[int]
            Positions that need detection. Empty list means skip all.
        """
        if force:
            return list(global_position_indices)

        # More efficient: convert to set directly
        existing_positions = set(
            session.exec(
                select(FOV.position_index)
                .join(ROI)
                .where(
                    ROI.detection_settings_id == detection_settings_id,
                    FOV.position_index.in_(global_position_indices),  # type: ignore
                )
                .distinct()
            ).all()
        )

        positions_needing_detection = [
            p for p in global_position_indices if p not in existing_positions
        ]

        if not positions_needing_detection:
            cali_logger.info(
                "⚠️ Detection already exists for all positions with "
                f"DetectionSettings ID {detection_settings_id}. Skipping detection."
            )
            return []

        if existing_positions:
            cali_logger.info(
                f"⚠️  Detection exists for {len(existing_positions)} position(s) "
                f"but missing for {len(positions_needing_detection)} position(s): "
                f"{positions_needing_detection}. "
                "Running detection for missing positions."
            )

        return positions_needing_detection

    def _get_positions_for_extraction(
        self,
        session: Session,
        detection_settings_id: int,
        extraction_settings_id: int,
        global_position_indices: Sequence[int],
        force: bool = False,
    ) -> list[int]:
        """Get positions that need extraction.

        Returns positions that either:
        - Don't have Traces with this extraction settings yet, OR
        - force=True (re-run all)

        Parameters
        ----------
        session : Session
            Database session
        detection_settings_id : int
            Detection settings ID to check
        extraction_settings_id : int
            Extraction settings ID to check
        global_position_indices : Sequence[int]
            Positions to check
        force : bool
            If True, returns all positions regardless of existing results.

        Returns
        -------
        list[int]
            Positions that need extraction. Empty list means skip all.
        """
        if force:
            return list(global_position_indices)

        # Optimize by using set directly
        existing_positions = set(
            session.exec(
                select(FOV.position_index)
                .join(ROI)
                .join(Traces)
                .where(
                    ROI.detection_settings_id == detection_settings_id,
                    Traces.analysis_result_id.in_(  # type: ignore
                        select(CaliResult.id).where(
                            CaliResult.extraction_settings_id == extraction_settings_id
                        )
                    ),
                    FOV.position_index.in_(global_position_indices),  # type: ignore
                )
                .distinct()
            ).all()
        )

        positions_needing_extraction = [
            p for p in global_position_indices if p not in existing_positions
        ]

        # Don't log here if we're just checking for analysis-only mode
        # The caller will log more detailed information about the split
        if not positions_needing_extraction and not force:
            # Still return empty list, but don't log yet
            # This allows the caller to provide context about whether
            # analysis will run on these positions
            return []

        if existing_positions and positions_needing_extraction:
            cali_logger.info(
                f"⚠️ Extraction exists for {len(existing_positions)} position(s) "
                f"but missing for {len(positions_needing_extraction)} position(s): "
                f"{positions_needing_extraction}. "
                "Running extraction for missing positions."
            )

        return positions_needing_extraction

    def _get_positions_for_analysis(
        self,
        session: Session,
        detection_settings_id: int,
        extraction_settings_id: int,
        analysis_settings_id: int,
        global_position_indices: Sequence[int],
        force: bool = False,
    ) -> list[int]:
        """Get positions that need analysis.

        Returns positions that either:
        - Don't have analysis results yet with this combination of settings, OR
        - force=True (re-run all)

        Parameters
        ----------
        session : Session
            Database session
        detection_settings_id : int
            Detection settings ID to check
        extraction_settings_id : int
            Extraction settings ID to check (analysis depends on extraction)
        analysis_settings_id : int
            Analysis settings ID to check
        global_position_indices : Sequence[int]
            Positions to check
        force : bool
            If True, returns all positions regardless of existing results.

        Returns
        -------
        list[int]
            Positions that need analysis. Empty list means skip all.
        """
        if force:
            return list(global_position_indices)

        # Optimize by using set directly
        # Build subquery for CaliResult filtering
        result_subquery = select(CaliResult.id).where(
            CaliResult.extraction_settings_id == extraction_settings_id,
            CaliResult.analysis_settings_id == analysis_settings_id,
        )

        existing_positions = set(
            session.exec(
                select(FOV.position_index)
                .join(ROI)
                .join(Traces)
                .where(
                    ROI.detection_settings_id == detection_settings_id,
                    Traces.analysis_result_id.in_(result_subquery),  # type: ignore
                    FOV.position_index.in_(global_position_indices),  # type: ignore
                )
                .distinct()
            ).all()
        )

        positions_needing_analysis = [
            p for p in global_position_indices if p not in existing_positions
        ]

        if not positions_needing_analysis:
            cali_logger.info(
                f"⚠️ Analysis already exists for all positions with "
                f"DetectionSettings ID {detection_settings_id}, "
                f"ExtractionSettings ID {extraction_settings_id}, and "
                f"AnalysisSettings ID {analysis_settings_id}. Skipping analysis."
            )
            return []

        if existing_positions:
            cali_logger.info(
                f"⚠️ Analysis exists for {len(existing_positions)} position(s) "
                f"but missing for {len(positions_needing_analysis)} position(s): "
                f"{positions_needing_analysis}. Running analysis for missing positions."
            )

        return positions_needing_analysis

    def _setup_database(
        self,
        db_path: Path,
        experiment: Experiment,
        overwrite: bool = False,
    ) -> None:
        """Setup database - create if doesn't exist, validate if it does."""
        # Determine database path
        # Database doesn't exist - create it
        if not Path(db_path).exists():
            cali_logger.info(f"💾 Creating new database at {db_path}")
            save_experiment_to_database(
                experiment, db_path.parent, database_name=db_path.name
            )
            # experiment.id is now set by save_experiment_to_database
            return

        # Database exists
        if overwrite:
            cali_logger.info(f"🔄 Overwriting existing database at {db_path}")
            save_experiment_to_database(
                experiment, db_path.parent, database_name=db_path.name, overwrite=True
            )
            # experiment.id is now set by save_experiment_to_database
        else:
            # Validate experiment ID matches database
            engine = create_engine(
                f"sqlite:///{db_path}",
                echo=False,
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    from cali.sqlmodel._model import Experiment

                    db_exp = cast(
                        "Experiment", session.exec(select(Experiment)).first()
                    )
                    # Check if they match using __eq__ (compares name)
                    if experiment != db_exp:
                        msg = (
                            f"The provided Experiment (name='{experiment.name}') "
                            f"does not match the one in the database "
                            f"(name='{db_exp.name}'). "
                            f"To run a different experiment, specify a unique "
                            f"`database_name` parameter in the run() method. "
                            f"To replace the existing database, set `overwrite=True`."
                        )
                        cali_logger.error(msg)
                        raise ValueError(msg)
                    # Set the experiment ID from the database
                    experiment.id = db_exp.id
            finally:
                engine.dispose(close=True)

    def _get_or_create_detection_settings(
        self, session: Session, detection_settings: DetectionSettings | int
    ) -> DetectionSettings:
        """Get existing or create new DetectionSettings in database.

        This implements settings deduplication - if identical settings exist,
        reuse them via pointer (foreign key) instead of creating duplicates.
        """
        from sqlalchemy.orm import exc as orm_exc

        from cali.sqlmodel._model import DetectionSettings

        if isinstance(detection_settings, int):
            # Load existing settings by ID
            existing = session.get(DetectionSettings, detection_settings)
            if existing is None:
                msg = (
                    f"DetectionSettings with ID {detection_settings} not found "
                    "in database."
                )
                cali_logger.error(msg)
                raise ValueError(msg)
            assert isinstance(existing, DetectionSettings)
            cali_logger.info(
                f"♻️ Reusing existing DetectionSettings ID {existing.id} "
                f"(method: {existing.method})"
            )
            return existing

        # Check if the object is detached and needs reattaching
        try:
            settings_id = detection_settings.id
        except orm_exc.DetachedInstanceError:
            # Object was detached - merge it back into session
            detection_settings = session.merge(detection_settings)
            assert isinstance(detection_settings, DetectionSettings)
            cali_logger.info(
                f"♻️ Reattached DetectionSettings ID {detection_settings.id} "
                f"(method: {detection_settings.method})"
            )
            return detection_settings

        if settings_id is None:
            # Check if identical settings already exist
            all_settings: list[DetectionSettings] = list(
                session.exec(select(DetectionSettings)).all()
            )
            for candidate in all_settings:
                if detection_settings == candidate:
                    cali_logger.info(
                        f"♻️ Reusing existing DetectionSettings ID {candidate.id} "
                        f"(method: {candidate.method})"
                    )
                    return candidate

            # New settings - create them
            session.add(detection_settings)
            session.commit()
            session.refresh(detection_settings)
            cali_logger.info(
                f"⚙️ Created new DetectionSettings ID {detection_settings.id} "
                f"(method: {detection_settings.method})"
            )
            return detection_settings
        else:
            # Settings has ID - check if exists in database
            existing = session.get(DetectionSettings, detection_settings.id)
            if existing is not None:
                assert isinstance(existing, DetectionSettings)
                cali_logger.info(
                    f"♻️ Reusing existing DetectionSettings ID {existing.id} "
                    f"(method: {existing.method})"
                )
                return existing

            # ID doesn't exist - create it
            session.add(detection_settings)
            session.commit()
            session.refresh(detection_settings)
            cali_logger.info(
                f"⚙️ Created new DetectionSettings ID {detection_settings.id} "
                f"(method: {detection_settings.method})"
            )
            return detection_settings

    def _get_or_create_extraction_settings(
        self, session: Session, extraction_settings: ExtractionSettings | int
    ) -> ExtractionSettings:
        """Get existing or create new ExtractionSettings in database.

        This implements settings deduplication - if identical settings exist,
        reuse them via pointer (foreign key) instead of creating duplicates.
        """
        from cali.sqlmodel._model import ExtractionSettings

        if isinstance(extraction_settings, int):
            # Load existing settings by ID
            existing = session.get(ExtractionSettings, extraction_settings)
            if existing is None:
                msg = (
                    f"ExtractionSettings ID {extraction_settings} not found in database"
                )
                cali_logger.error(msg)
                raise ValueError(msg)
            assert isinstance(existing, ExtractionSettings)
            cali_logger.info(f"♻️ Reusing existing ExtractionSettings ID {existing.id}")
            return existing

        elif extraction_settings.id is None:
            # Check if identical settings already exist
            all_settings: list[ExtractionSettings] = list(
                session.exec(select(ExtractionSettings)).all()
            )
            for candidate in all_settings:
                if extraction_settings == candidate:
                    cali_logger.info(
                        f"♻️ Reusing existing ExtractionSettings ID {candidate.id}"
                    )
                    return candidate

            # New settings - create them
            session.add(extraction_settings)
            session.commit()
            session.refresh(extraction_settings)
            cali_logger.info(
                f"⚙️ Created new ExtractionSettings ID {extraction_settings.id}"
            )
            return extraction_settings
        else:
            # Settings has ID - check if exists in database
            existing = session.get(ExtractionSettings, extraction_settings.id)
            if existing is not None:
                assert isinstance(existing, ExtractionSettings)
                cali_logger.info(
                    f"♻️ Reusing existing ExtractionSettings ID {existing.id}"
                )
                return existing

            # ID doesn't exist - create it
            session.add(extraction_settings)
            session.commit()
            session.refresh(extraction_settings)
            cali_logger.info(
                f"⚙️ Created new ExtractionSettings ID {extraction_settings.id}"
            )
            return extraction_settings

    def _get_or_create_analysis_settings(
        self, session: Session, analysis_settings: AnalysisSettings | int
    ) -> AnalysisSettings:
        """Get existing or create new AnalysisSettings in database.

        This implements settings deduplication - if identical settings exist,
        reuse them via pointer (foreign key) instead of creating duplicates.
        """
        from cali.sqlmodel._model import AnalysisSettings

        if isinstance(analysis_settings, int):
            # Load existing settings by ID
            existing = session.get(AnalysisSettings, analysis_settings)
            if existing is None:
                msg = (
                    f"AnalysisSettings with ID {analysis_settings} not found "
                    "in database."
                )
                cali_logger.error(msg)
                raise ValueError(msg)
            assert isinstance(existing, AnalysisSettings)
            cali_logger.info(f"♻️ Reusing existing AnalysisSettings ID {existing.id}")
            return existing

        elif analysis_settings.id is None:
            # Check if identical settings already exist
            all_settings: list[AnalysisSettings] = list(
                session.exec(select(AnalysisSettings)).all()
            )
            for candidate in all_settings:
                if analysis_settings == candidate:
                    cali_logger.info(
                        f"♻️ Reusing existing AnalysisSettings ID {candidate.id}"
                    )
                    return candidate

            # New settings - merge and commit
            analysis_settings = session.merge(analysis_settings)
            assert isinstance(analysis_settings, AnalysisSettings)
            session.commit()
            session.refresh(analysis_settings)
            cali_logger.info(
                f"⚙️ Created new AnalysisSettings ID {analysis_settings.id}"
            )
            return analysis_settings
        else:
            # Settings has ID - merge to reattach
            analysis_settings = session.merge(analysis_settings)
            assert isinstance(analysis_settings, AnalysisSettings)
            cali_logger.info(
                f"♻️ Reusing existing AnalysisSettings ID {analysis_settings.id}"
            )
            return analysis_settings

    def _run_detection(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
    ) -> Generator[FOV, None, None]:
        """Run detection using DetectionRunner (pure computation).

        Yields FOV results from DetectionRunner.
        """
        cali_logger.info("🔍 Running Detection...")
        yield from self._detection_runner.run(
            dataset=dataset,
            detection_settings=detection_settings,
            global_position_indices=global_position_indices,
            as_generator=True,
        )

    def _run_extraction(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        extraction_settings: ExtractionSettings,
        analysis_settings: AnalysisSettings | None,
        fovs: Iterable[FOV],
    ) -> Generator[FOV, None, None]:
        """Run extraction using ExtractionRunner.

        Yields FOV results from ExtractionRunner using threaded processing.

        Parameters
        ----------
        dataset : TensorstoreZarrReader | OMEZarrReader
            Dataset reader
        extraction_settings : ExtractionSettings
            Extraction configuration (neuropil, dff, deconvolution)
        analysis_settings : AnalysisSettings | None
            Analysis configuration (peak detection, thresholds)
            If None, only extraction is performed
        fovs : Iterable[FOV]
            FOVs with ROIs to analyze (from detection or loaded from DB)
        """
        msg = "📈 Running Extraction"
        msg = msg + (" and 📊 Analysis..." if analysis_settings is not None else "...")
        cali_logger.info(msg)
        yield from self._extraction_runner.run(
            dataset=dataset,
            extraction_settings=extraction_settings,
            fovs=fovs,
            analysis_settings=analysis_settings,
            as_generator=True,
        )

    def _process_fov_results(
        self,
        fov: FOV,
        session: Session,
        analysis_result_id: int | None,
        include_traces: bool,
    ) -> None:
        """Process FOV results by moving temporary data to permanent collections.

        Parameters
        ----------
        fov : FOV
            The FOV with temporary result data
        session : Session
            Database session for adding FOV-level analysis
        analysis_result_id : int | None
            The CaliResult ID to associate with the results
        include_traces : bool
            Whether to process traces (True for extraction, False for analysis-only)
        """
        if analysis_result_id is None:
            return

        for roi in fov.rois:
            # Process traces (only for extraction path)
            if include_traces and hasattr(roi, "_new_traces"):
                for trace in roi._new_traces:
                    trace.analysis_result_id = analysis_result_id
                    roi.traces_history.append(trace)
                delattr(roi, "_new_traces")

            # Process ROI-level analysis (both extraction and analysis-only)
            if hasattr(roi, "_new_data_analysis"):
                for data_analysis in roi._new_data_analysis:
                    data_analysis.analysis_result_id = analysis_result_id
                    roi.data_analysis_history.append(data_analysis)
                delattr(roi, "_new_data_analysis")

        # Process FOV-level analysis (both extraction and analysis-only)
        if hasattr(fov, "_new_fov_analysis"):
            for fov_analysis in fov._new_fov_analysis:
                fov_analysis.analysis_result_id = analysis_result_id
                fov_analysis.fov_id = fov.id
                session.add(fov_analysis)
            delattr(fov, "_new_fov_analysis")

    def _run_analysis_only(
        self,
        analysis_settings: AnalysisSettings,
        fovs: Iterable[FOV],
    ) -> Generator[FOV, None, None]:
        """Run analysis only on FOVs that already have extraction results.

        Parameters
        ----------
        analysis_settings : AnalysisSettings
            Analysis configuration (peak detection, thresholds)
        fovs : Iterable[FOV]
            FOVs with ROIs and existing traces to analyze
        """
        from cali.analysis._fov_analysis import compute_fov_analysis

        cali_logger.info("📊 Running Analysis (using existing extraction)...")
        for fov in fovs:
            fov_analysis = compute_fov_analysis(fov, analysis_settings)
            if fov_analysis is not None:
                if not hasattr(fov, "_new_fov_analysis"):
                    fov._new_fov_analysis = []
                fov._new_fov_analysis.append(fov_analysis)
            yield fov

    def _load_fovs_from_db(
        self,
        session: Session,
        detection_settings_id: int,
        position_indices: Sequence[int],
    ) -> list[FOV]:
        """Load FOVs with ROIs from database.

        Parameters
        ----------
        session : Session
            Database session
        detection_settings_id : int
            Detection settings ID to filter ROIs
        position_indices : Sequence[int]
            Position indices to load

        Returns
        -------
        list[FOV]
            FOVs with ROIs and masks eagerly loaded from database
        """
        from sqlalchemy.orm import joinedload

        # Load FOVs with eager loading of related data
        # NOTE: Only load ROI masks - we don't need historical traces/analysis DATA
        # since we're creating new ones. However, we DO need to load the collection
        # structure itself (empty lists) to avoid lazy load errors when appending.
        fovs = (
            session.exec(
                select(FOV)
                .where(FOV.position_index.in_(position_indices))  # type: ignore
                .options(
                    joinedload(FOV.rois).joinedload(ROI.roi_mask),
                )
            )
            .unique()
            .all()
        )

        # joinedload still loads all ROIs due to relationship behavior
        # Still need to filter the ROI collection itself
        for fov in fovs:
            fov.rois = [
                roi
                for roi in fov.rois
                if roi.detection_settings_id == detection_settings_id
            ]
            # Force-load collection structures before detaching
            # This just loads empty lists, not the actual data
            for roi in fov.rois:
                _ = roi.traces_history
                _ = roi.data_analysis_history
            _ = fov.fov_analysis_history

        filtered_fovs = [fov for fov in fovs if fov.rois]

        return filtered_fovs

    def _create_or_update_analysis_result(
        self,
        session: Session,
        experiment_id: int,
        detection_settings_id: int,
        extraction_settings_id: int | None,
        analysis_settings_id: int | None,
        positions_detected: list[int] | None = None,
        positions_extracted: list[int] | None = None,
        positions_analyzed: list[int] | None = None,
    ) -> tuple[int, bool]:
        """Create or update a CaliResult entry with progressive stage tracking.

        Handles progressive analysis stages:
        1. If exact match exists (same detection, extraction, analysis settings),
           merge positions
        2. If partial match exists (same detection/extraction but less complete),
           upgrade the existing result with new settings
        3. Otherwise, create new result and clean up superseded ones

        Parameters
        ----------
        session : Session
            Database session
        experiment_id : int
            Experiment ID
        detection_settings_id : int
            Detection settings ID
        extraction_settings_id : int | None
            Extraction settings ID (None for detection-only)
        analysis_settings_id : int | None
            Analysis settings ID (None for extraction-only or detection-only)
        positions_detected : list[int] | None
            Positions with detection results (ROIs)
        positions_extracted : list[int] | None
            Positions with extraction results (traces)
        positions_analyzed : list[int] | None
            Positions with full analysis results

        Returns
        -------
            Tuple of (result_id, was_created) where was_created=True for new results.
        """
        # Ensure at least one position list is provided
        if (
            positions_detected is None
            and positions_extracted is None
            and positions_analyzed is None
        ):
            raise ValueError(
                "At least one of positions_detected, positions_extracted, "
                "or positions_analyzed must be provided"
            )

        # First, check for exact match with all settings
        query = select(CaliResult).where(
            CaliResult.experiment == experiment_id,
            CaliResult.detection_settings_id == detection_settings_id,
        )

        if extraction_settings_id is None:
            query = query.where(CaliResult.extraction_settings_id.is_(None))  # type: ignore
        else:
            query = query.where(
                CaliResult.extraction_settings_id == extraction_settings_id
            )

        if analysis_settings_id is None:
            query = query.where(CaliResult.analysis_settings_id.is_(None))  # type: ignore
        else:
            query = query.where(CaliResult.analysis_settings_id == analysis_settings_id)

        exact_match = session.exec(query).first()

        if exact_match:
            assert isinstance(exact_match, CaliResult)
            # Update existing result by merging positions for each stage
            if positions_detected is not None:
                old = set(exact_match.positions_detected or [])
                new = set(positions_detected)
                exact_match.positions_detected = sorted(old | new)

            if positions_extracted is not None:
                old = set(exact_match.positions_extracted or [])
                new = set(positions_extracted)
                exact_match.positions_extracted = sorted(old | new)

            if positions_analyzed is not None:
                old = set(exact_match.positions_analyzed or [])
                new = set(positions_analyzed)
                exact_match.positions_analyzed = sorted(old | new)

            exact_match.last_modified = datetime.now()

            session.add(exact_match)
            session.commit()
            session.refresh(exact_match)

            result_type = self._get_result_type(
                extraction_settings_id, analysis_settings_id
            )
            cali_logger.info(
                f"📝 Updated {result_type} CaliResult ID {exact_match.id} "
                f"(DetectionSettings={detection_settings_id}, "
                f"ExtractionSettings={extraction_settings_id}, "
                f"AnalysisSettings={analysis_settings_id}, "
                f"detected={exact_match.positions_detected}, "
                f"extracted={exact_match.positions_extracted}, "
                f"analyzed={exact_match.positions_analyzed})"
            )
            assert exact_match.id is not None
            return (exact_match.id, False)

        # Check for a less-complete result that can be upgraded
        # (same detection/extraction but missing analysis)
        if analysis_settings_id is not None and extraction_settings_id is not None:
            upgradeable_result = session.exec(
                select(CaliResult).where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings_id == detection_settings_id,
                    CaliResult.extraction_settings_id == extraction_settings_id,
                    CaliResult.analysis_settings_id.is_(None),  # type: ignore
                )
            ).first()

            if upgradeable_result:
                assert isinstance(upgradeable_result, CaliResult)
                # Upgrade the existing result with analysis settings and merge positions
                if positions_detected is not None:
                    old = set(upgradeable_result.positions_detected or [])
                    new = set(positions_detected)
                    upgradeable_result.positions_detected = sorted(old | new)

                if positions_extracted is not None:
                    old = set(upgradeable_result.positions_extracted or [])
                    new = set(positions_extracted)
                    upgradeable_result.positions_extracted = sorted(old | new)

                if positions_analyzed is not None:
                    old = set(upgradeable_result.positions_analyzed or [])
                    new = set(positions_analyzed)
                    upgradeable_result.positions_analyzed = sorted(old | new)

                upgradeable_result.analysis_settings_id = analysis_settings_id

                upgradeable_result.last_modified = datetime.now()

                session.add(upgradeable_result)
                session.commit()
                session.refresh(upgradeable_result)

                cali_logger.info(
                    f"⬆️ Upgraded CaliResult ID {upgradeable_result.id} "
                    f"with AnalysisSettings={analysis_settings_id}, "
                    f"analyzed={upgradeable_result.positions_analyzed}"
                )
                assert upgradeable_result.id is not None
                return (upgradeable_result.id, True)  # True: treat like creation

        # Check for detection-only result to update when running detection-only
        if extraction_settings_id is None and analysis_settings_id is None:
            # First, check for ANY existing result with same detection settings
            # (regardless of extraction/analysis settings)
            all_results_with_detection = session.exec(
                select(CaliResult)
                .where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings_id == detection_settings_id,
                )
                .order_by(
                    # Prefer more complete results
                    # (analysis > extraction > detection-only)
                    CaliResult.analysis_settings_id.is_(None),  # type: ignore
                    CaliResult.extraction_settings_id.is_(None),  # type: ignore
                )
            ).all()

            if len(all_results_with_detection) > 1:
                # Multiple runs exist with the same detection but different
                # extraction/analysis settings. User must disambiguate by
                # specifying extraction and/or analysis settings.

                # Check what varies across results
                extraction_ids = {
                    r.extraction_settings_id
                    for r in all_results_with_detection
                    if r.extraction_settings_id is not None
                }
                analysis_ids = {
                    r.analysis_settings_id
                    for r in all_results_with_detection
                    if r.analysis_settings_id is not None
                }

                if analysis_ids:
                    # Different analysis settings exist
                    msg = (
                        f"Multiple runs exist with the same detection settings "
                        f"(ID {detection_settings_id}) but different analysis "
                        f"settings. Please specify both extraction_settings and "
                        f"analysis_settings to indicate which run the new "
                        f"detected positions should be added to."
                    )
                elif extraction_ids:
                    # Different extraction settings exist (no analysis)
                    msg = (
                        f"Multiple runs exist with the same detection settings "
                        f"(ID {detection_settings_id}) but different extraction "
                        f"settings. Please specify extraction_settings to "
                        f"indicate which run the new detected positions should "
                        f"be added to."
                    )
                else:
                    # This shouldn't happen, but handle gracefully
                    msg = (
                        f"Multiple runs exist with detection settings ID "
                        f"{detection_settings_id}. Please specify extraction_settings "
                        f"and/or analysis_settings to disambiguate."
                    )

                cali_logger.error(msg)
                raise ValueError(msg)

            if all_results_with_detection:
                any_result_with_detection = all_results_with_detection[0]
                assert isinstance(any_result_with_detection, CaliResult)
                # Update the most complete existing result's detected positions
                if positions_detected is not None:
                    old = set(any_result_with_detection.positions_detected or [])
                    new = set(positions_detected)
                    any_result_with_detection.positions_detected = sorted(old | new)

                any_result_with_detection.last_modified = datetime.now()

                session.add(any_result_with_detection)
                session.commit()
                session.refresh(any_result_with_detection)

                result_type = self._get_result_type(
                    any_result_with_detection.extraction_settings_id,
                    any_result_with_detection.analysis_settings_id,
                )
                cali_logger.info(
                    f"📝 Updated {result_type} CaliResult ID "
                    f"{any_result_with_detection.id} with new detected positions "
                    f"(detected={any_result_with_detection.positions_detected})"
                )
                assert any_result_with_detection.id is not None
                return (any_result_with_detection.id, False)  # False: not newly created

        # Check for result with same detection/extraction (regardless of analysis)
        # when we're running detection+extraction without analysis
        if extraction_settings_id is not None and analysis_settings_id is None:
            # First, check for detection-only result to upgrade
            upgradeable_result = session.exec(
                select(CaliResult).where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings_id == detection_settings_id,
                    CaliResult.extraction_settings_id.is_(None),  # type: ignore
                    CaliResult.analysis_settings_id.is_(None),  # type: ignore
                )
            ).first()

            if upgradeable_result:
                assert isinstance(upgradeable_result, CaliResult)
                # Upgrade result with extraction settings and merge positions
                if positions_detected is not None:
                    old = set(upgradeable_result.positions_detected or [])
                    new = set(positions_detected)
                    upgradeable_result.positions_detected = sorted(old | new)

                if positions_extracted is not None:
                    old = set(upgradeable_result.positions_extracted or [])
                    new = set(positions_extracted)
                    upgradeable_result.positions_extracted = sorted(old | new)

                if positions_analyzed is not None:
                    old = set(upgradeable_result.positions_analyzed or [])
                    new = set(positions_analyzed)
                    upgradeable_result.positions_analyzed = sorted(old | new)

                upgradeable_result.extraction_settings_id = extraction_settings_id

                upgradeable_result.last_modified = datetime.now()

                session.add(upgradeable_result)
                session.commit()
                session.refresh(upgradeable_result)

                cali_logger.info(
                    f"⬆️ Upgraded CaliResult ID {upgradeable_result.id} "
                    f"with ExtractionSettings={extraction_settings_id}, "
                    f"extracted={upgradeable_result.positions_extracted}"
                )
                assert upgradeable_result.id is not None
                return (upgradeable_result.id, True)  # True: treat like creation

            # Second, check for existing result with same detection+extraction
            # (may have analysis settings, but we're just adding positions)
            compatible_result = session.exec(
                select(CaliResult).where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings_id == detection_settings_id,
                    CaliResult.extraction_settings_id == extraction_settings_id,
                )
            ).first()

            if compatible_result:
                assert isinstance(compatible_result, CaliResult)
                # Update positions for each stage (don't change analysis settings)
                if positions_detected is not None:
                    old = set(compatible_result.positions_detected or [])
                    new = set(positions_detected)
                    compatible_result.positions_detected = sorted(old | new)

                if positions_extracted is not None:
                    old = set(compatible_result.positions_extracted or [])
                    new = set(positions_extracted)
                    compatible_result.positions_extracted = sorted(old | new)

                if positions_analyzed is not None:
                    old = set(compatible_result.positions_analyzed or [])
                    new = set(positions_analyzed)
                    compatible_result.positions_analyzed = sorted(old | new)

                compatible_result.last_modified = datetime.now()

                session.add(compatible_result)
                session.commit()
                session.refresh(compatible_result)

                result_type = self._get_result_type(
                    extraction_settings_id, compatible_result.analysis_settings_id
                )
                cali_logger.info(
                    f"⬆️ Updated {result_type} CaliResult ID "
                    f"{compatible_result.id} with new positions "
                    f"(extracted={compatible_result.positions_extracted})"
                )
                assert compatible_result.id is not None
                return (compatible_result.id, False)  # False: not newly created

        # No upgradeable result found - create new one
        # First, delete any detection-only result that will be superseded
        if analysis_settings_id is not None:
            detection_only_result = session.exec(
                select(CaliResult).where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings_id == detection_settings_id,
                    CaliResult.extraction_settings_id.is_(None),  # type: ignore
                    CaliResult.analysis_settings_id.is_(None),  # type: ignore
                )
            ).first()

            if detection_only_result:
                cali_logger.info(
                    f"🗑️  Removing detection-only CaliResult ID "
                    f"{detection_only_result.id} (superseded by analysis result)"
                )
                session.delete(detection_only_result)
                session.commit()

        # Create new result
        result = CaliResult(
            experiment=experiment_id,
            detection_settings_id=detection_settings_id,
            extraction_settings_id=extraction_settings_id,
            analysis_settings_id=analysis_settings_id,
            positions_detected=positions_detected,
            positions_extracted=positions_extracted,
            positions_analyzed=positions_analyzed,
        )
        session.add(result)
        session.commit()
        session.refresh(result)

        result_type = self._get_result_type(
            extraction_settings_id, analysis_settings_id
        )
        cali_logger.info(
            f"⚙️ Created {result_type} CaliResult ID {result.id} "
            f"(DetectionSettings={detection_settings_id}, "
            f"ExtractionSettings={extraction_settings_id}, "
            f"AnalysisSettings={analysis_settings_id}, "
            f"detected={positions_detected}, extracted={positions_extracted}, "
            f"analyzed={positions_analyzed})"
        )
        assert result.id is not None
        return (result.id, True)

    def _get_result_type(
        self, extraction_settings_id: int | None, analysis_settings_id: int | None
    ) -> str:
        """Get descriptive type of CaliResult based on settings."""
        if analysis_settings_id is not None:
            return "full analysis"
        elif extraction_settings_id is not None:
            return "detection+extraction"
        else:
            return "detection-only"
