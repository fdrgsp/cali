"""Unified runner interface for detection and analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import event
from sqlmodel import Session, create_engine, select

from cali._constants import DEFAULT_CALI_DB_NAME, CorrelationDataType, TraceDataType
from cali.analysis import AnalysisRunner
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
        """Cancel detection, extraction, and analysis processes."""
        self._detection_runner.cancel()
        self._extraction_runner.cancel()
        if hasattr(self, "_analysis_runner"):
            self._analysis_runner.cancel()

    def run(
        self,
        experiment: Experiment,
        dataset_path: str | Path | None,
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
        dataset_path: str | Path | None,
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

        # 0. Make sure data are ready (can be None for analysis-only mode)
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader | None = (
            None
        )
        if dataset_path is not None:
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
            if dataset_path is not None:
                output_path = Path(dataset_path).parent
            else:
                msg = "❌ output_path is required when dataset_path is None."
                cali_logger.error(msg)
                raise ValueError(msg)
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
                    msg = (
                        "extraction_settings is required when "
                        "analysis_settings is provided"
                    )
                    cali_logger.error(msg)
                    raise ValueError(msg)

                # Narrow types to actual settings objects
                extraction_settings_obj: ExtractionSettings | None = None
                analysis_settings_obj: AnalysisSettings | None = None

                if extraction_settings is not None:
                    extraction_settings_obj = self._get_or_create_extraction_settings(
                        session, extraction_settings
                    )
                    extraction_settings_id = extraction_settings_obj.id
                    extraction_threads = extraction_settings_obj.threads
                    if extraction_settings_id is None:  # pragma: no cover
                        msg = "ExtractionSettings must have an ID after persistence."
                        cali_logger.error(msg)
                        raise ValueError(msg)
                else:
                    extraction_settings_id = None
                    extraction_threads = None

                if analysis_settings is not None:
                    analysis_settings_obj = self._get_or_create_analysis_settings(
                        session, analysis_settings
                    )
                    analysis_settings_id = analysis_settings_obj.id
                    if analysis_settings_id is None:  # pragma: no cover
                        msg = "AnalysisSettings must have an ID after persistence."
                        cali_logger.error(msg)
                        raise ValueError(msg)
                else:
                    analysis_settings_id = None

                det_id = detection_settings.id
                if det_id is None:  # pragma: no cover
                    msg = "DetectionSettings must have an ID after persistence."
                    cali_logger.error(msg)
                    raise ValueError(msg)

                # 4. Determine which positions need detection
                # Track whether user explicitly provided position indices
                user_provided_positions = global_position_indices is not None

                if global_position_indices is None:
                    if dataset is None or dataset.sequence is None:
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

                # 5. Run detection if needed
                analysis_result_id: int | None = None
                if positions_for_detection and dataset is None:
                    msg = (
                        "Dataset is required for detection. "
                        "Provide a data path or use analysis-only mode."
                    )
                    cali_logger.error(msg)
                    raise ValueError(msg)
                if positions_for_detection:
                    yield "🔍 Running Detection..."
                    cancelled = yield from self._run_detection_phase(
                        session=session,
                        experiment=experiment,
                        dataset=dataset,  # type: ignore
                        detection_settings=detection_settings,
                        det_id=det_id,
                        positions_for_detection=positions_for_detection,
                        detection_only=(
                            extraction_settings is None and analysis_settings is None
                        ),
                    )
                    if cancelled:
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

                    # Determine what type of run this is and display appropriate message
                    is_analysis_only = (
                        analysis_settings_obj is not None
                        and len(positions_need_analysis) > 0
                        and positions_only_analysis == positions_need_analysis
                    )

                    if is_analysis_only:
                        yield "📊 Running Analysis..."
                        cali_logger.info(
                            f"⚠️ Extraction already exists for all positions "
                            f"with DetectionSettings ID {det_id} and "
                            f"ExtractionSettings ID {extraction_settings_id}. "
                            "Running analysis only on these positions."
                        )
                    else:
                        yield "📈 Running Extraction" + (
                            " and 📊 Analysis..." if analysis_settings_obj else "..."
                        )
                        # Log if some positions are analysis-only
                        if positions_only_analysis:
                            positions_list = sorted(positions_only_analysis)
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
                    if extraction_threads is None:  # pragma: no cover
                        msg = "extraction_threads must be set when running extraction."
                        cali_logger.error(msg)
                        raise ValueError(msg)
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
                        if fovs_need_extraction:
                            if dataset is None:
                                msg = (
                                    "Dataset is required for extraction. "
                                    "Provide a data path or use "
                                    "analysis-only mode."
                                )
                                cali_logger.error(msg)
                                raise ValueError(msg)
                            for fov in self._run_extraction(
                                dataset,
                                extraction_settings_obj,
                                analysis_settings_obj,
                                fovs=fovs_need_extraction,
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
                                self._commit_fov_batch(
                                    fov,
                                    session,
                                    experiment,
                                    fov_count,
                                    len(positions_to_process),
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
                                    source_extraction_settings_id=extraction_settings_id,
                                    source_detection_settings_id=det_id,
                                )
                                fov_count += 1
                                batch_fov_count += 1
                                yield "PROGRESS:UPDATE"
                                self._commit_fov_batch(
                                    fov,
                                    session,
                                    experiment,
                                    fov_count,
                                    len(positions_to_process),
                                )
                                positions_processed.append(fov.position_index)

                        # Commit any remaining in this batch and clear memory
                        self._commit_remaining_fovs(batch_fov_count, session)

                        # Expunge FOVs to free memory
                        for fov in batch_fovs:
                            if fov in session:
                                session.expunge(fov)

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
                                result.positions_extracted = self._merge_positions(
                                    result.positions_extracted, positions_processed
                                )
                                if analysis_settings_id is not None:
                                    result.positions_analyzed = self._merge_positions(
                                        result.positions_analyzed, positions_processed
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

                        # Export multi-well aggregated data if requested
                        if export_correlations.get("Multi-Well Aggregated Data", False):
                            from cali.util._database_to_csv import (
                                export_multi_well_to_csv,
                            )

                            experiment_type = (
                                analysis_settings_obj.experiment_type
                                if analysis_settings_obj is not None
                                else None
                            )
                            yield "🗂️ Exporting multi-well aggregated data..."
                            export_multi_well_to_csv(
                                engine,
                                analysis_result_id,
                                self._db_path,
                                experiment_type=experiment_type,
                            )
                            # Also export PCA data
                            # from cali.util._database_to_csv import (
                            #     export_multi_well_pca_to_csv,
                            # )

                            # export_multi_well_pca_to_csv(
                            #     engine,
                            #     analysis_result_id,
                            #     self._db_path,
                            # )
                            yield "🗂️ Exported multi-well aggregated data"

        finally:
            cali_logger.info("🏁 Cali Run finished!")
            engine.dispose(close=True)

    # ==================== PHASE METHODS ====================

    def _run_detection_phase(
        self,
        session: Session,
        experiment: Experiment,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        detection_settings: DetectionSettings,
        det_id: int,
        positions_for_detection: list[int],
        detection_only: bool,
    ) -> Generator[str, None, bool]:
        """Run the detection phase, committing results to the database.

        Parameters
        ----------
        session : Session
            Database session.
        experiment : Experiment
            Experiment being processed.
        dataset : TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader
            Dataset to detect ROIs in.
        detection_settings : DetectionSettings
            Detection configuration.
        det_id : int
            Persisted DetectionSettings ID.
        positions_for_detection : list[int]
            Position indices that need detection.
        detection_only : bool
            True if no extraction/analysis will follow (creates a detection-
            only CaliResult).

        Returns
        -------
        bool
            True if the run was cancelled and the caller should stop.
        """
        detection_result_id: int | None = None
        positions_processed: list[int] = []
        total_rois_detected = 0

        # Create detection-only result if no extraction/analysis will follow
        detection_result_was_created = False
        if positions_for_detection and detection_only and experiment.id is not None:
            detection_result_id, detection_result_was_created = (
                self._create_or_update_analysis_result(
                    session=session,
                    experiment_id=experiment.id,
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
                total_rois_detected += len(fov.rois)
                fov_count += 1
                yield "PROGRESS:UPDATE"

                # Capture position index before expunging
                pos_idx = fov.position_index

                committed = self._commit_fov_batch(
                    fov,
                    session,
                    experiment,
                    fov_count,
                    len(positions_for_detection),
                    detection_settings_id=det_id,
                )
                if committed and fov in session:
                    session.expunge(fov)

                positions_processed.append(pos_idx)

            self._commit_remaining_fovs(fov_count, session)

            # Log detection completion
            if positions_processed:
                cali_logger.info(
                    f"✅ Detection committed: {total_rois_detected} "
                    f"ROIs across {len(positions_processed)} FOVs"
                )

            # Update detection-only result with actually completed positions
            if detection_result_id is not None:
                result = session.get(CaliResult, detection_result_id)
                if result:
                    if detection_result_was_created:
                        completed: list[int] = sorted(positions_processed)
                        result.positions_detected = completed
                    else:
                        completed = (
                            self._merge_positions(
                                result.positions_detected,
                                positions_processed,
                            )
                            or []
                        )
                        result.positions_detected = completed
                    session.add(result)
                    session.commit()
                    cali_logger.info(
                        f"📝 Updated CaliResult ID {detection_result_id} "
                        f"with completed detected positions: {completed}"
                    )

        # Check for cancellation after detection completes.
        # If cancelled and we were planning to run extraction/analysis,
        # create a detection-only result for the completed positions.
        if self._detection_runner._cancellation_event.is_set():
            if (
                positions_processed
                and detection_result_id is None
                and experiment.id is not None
            ):
                detection_result_id, _ = self._create_or_update_analysis_result(
                    session=session,
                    experiment_id=experiment.id,
                    detection_settings_id=det_id,
                    extraction_settings_id=None,
                    analysis_settings_id=None,
                    positions_detected=sorted(positions_processed),
                )
                session.commit()
                cali_logger.info(
                    f"📝 Created CaliResult ID {detection_result_id} for "
                    f"cancelled run with detected positions: "
                    f"{sorted(positions_processed)}"
                )
            return True  # cancelled

        return False  # not cancelled

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

    def _get_positions_needing_work(
        self,
        session: Session,
        query: Any,
        global_position_indices: Sequence[int],
        label: str,
        force: bool = False,
        *,
        skip_all_message: str | None = None,
    ) -> list[int]:
        """Return positions from *global_position_indices* not yet processed.

        Parameters
        ----------
        session : Session
            Database session.
        query : Select
            Query that returns `FOV.position_index` for already-processed
            positions.
        global_position_indices : Sequence[int]
            Full set of requested positions.
        label : str
            Human-readable phase name for log messages (e.g. "Detection").
        force : bool
            If True, return all positions regardless of existing results.
        skip_all_message : str | None
            If provided, log this message when all positions are already done.
            If None, return silently (useful when the caller logs instead).
        """
        if force:
            return list(global_position_indices)

        existing = set(session.exec(query).all())

        needed = [p for p in global_position_indices if p not in existing]

        if not needed:
            if skip_all_message:
                cali_logger.info(skip_all_message)
            return []

        if existing:
            cali_logger.info(
                f"⚠️ {label} exists for {len(existing)} position(s) "
                f"but missing for {len(needed)} position(s): {needed}. "
                f"Running {label.lower()} for missing positions."
            )

        return needed

    def _get_positions_for_detection(
        self,
        session: Session,
        detection_settings_id: int,
        global_position_indices: Sequence[int],
        force: bool = False,
    ) -> list[int]:
        """Get positions that need detection."""
        query = (
            select(FOV.position_index)
            .join(ROI)
            .where(
                ROI.detection_settings_id == detection_settings_id,
                FOV.position_index.in_(global_position_indices),  # type: ignore
            )
            .distinct()
        )
        return self._get_positions_needing_work(
            session,
            query,
            global_position_indices,
            "Detection",
            force,
            skip_all_message=(
                "⚠️ Detection already exists for all positions with "
                f"DetectionSettings ID {detection_settings_id}. Skipping detection."
            ),
        )

    def _get_positions_for_extraction(
        self,
        session: Session,
        detection_settings_id: int,
        extraction_settings_id: int,
        global_position_indices: Sequence[int],
        force: bool = False,
    ) -> list[int]:
        """Get positions that need extraction."""
        query = (
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
        )
        # Don't log "skip all" — caller provides context about analysis-only
        return self._get_positions_needing_work(
            session,
            query,
            global_position_indices,
            "Extraction",
            force,
        )

    def _get_positions_for_analysis(
        self,
        session: Session,
        detection_settings_id: int,
        extraction_settings_id: int,
        analysis_settings_id: int,
        global_position_indices: Sequence[int],
        force: bool = False,
    ) -> list[int]:
        """Get positions that need analysis."""
        result_subquery = select(CaliResult.id).where(
            CaliResult.extraction_settings_id == extraction_settings_id,
            CaliResult.analysis_settings_id == analysis_settings_id,
        )
        query = (
            select(FOV.position_index)
            .join(ROI)
            .join(Traces)
            .where(
                ROI.detection_settings_id == detection_settings_id,
                Traces.analysis_result_id.in_(result_subquery),  # type: ignore
                FOV.position_index.in_(global_position_indices),  # type: ignore
            )
            .distinct()
        )
        return self._get_positions_needing_work(
            session,
            query,
            global_position_indices,
            "Analysis",
            force,
            skip_all_message=(
                f"⚠️ Analysis already exists for all positions with "
                f"DetectionSettings ID {detection_settings_id}, "
                f"ExtractionSettings ID {extraction_settings_id}, and "
                f"AnalysisSettings ID {analysis_settings_id}. Skipping analysis."
            ),
        )

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

    def _resolve_settings(
        self,
        session: Session,
        settings: Any,
        model_class: type,
        label: str,
        *,
        use_merge: bool = False,
    ) -> Any:
        """Get existing or create new settings in database.

        Implements settings deduplication - if identical settings already exist,
        reuse them via pointer (foreign key) instead of creating duplicates.

        Parameters
        ----------
        session : Session
            Database session.
        settings : settings object or int
            Settings instance or existing settings ID.
        model_class : type
            The SQLModel class (DetectionSettings, ExtractionSettings, etc.).
        label : str
            Human-readable name for log messages.
        use_merge : bool
            If True, use `session.merge()` instead of `session.add()` when
            persisting. Required for AnalysisSettings which may reference
            objects from other sessions.
        """
        if isinstance(settings, int):
            existing = session.get(model_class, settings)
            if existing is None:
                msg = f"{label} with ID {settings} not found in database."
                cali_logger.error(msg)
                raise ValueError(msg)
            cali_logger.info(f"♻️ Reusing existing {label} ID {existing.id}")
            return existing

        if settings.id is None:
            # Content-based dedup: check if identical settings already exist
            for candidate in session.exec(select(model_class)).all():
                if settings == candidate:
                    cali_logger.info(f"♻️ Reusing existing {label} ID {candidate.id}")
                    return candidate

            # No match — persist new settings
            if use_merge:
                settings = session.merge(settings)
            else:
                session.add(settings)
            session.commit()
            session.refresh(settings)
            cali_logger.info(f"⚙️ Created new {label} ID {settings.id}")
            return settings

        # Settings already has an ID
        if use_merge:
            settings = session.merge(settings)
            cali_logger.info(f"♻️ Reusing existing {label} ID {settings.id}")
            return settings

        # Check DB for existing row with this ID
        existing = session.get(model_class, settings.id)
        if existing is not None:
            cali_logger.info(f"♻️ Reusing existing {label} ID {existing.id}")
            return existing

        # ID doesn't exist in DB — create it
        session.add(settings)
        session.commit()
        session.refresh(settings)
        cali_logger.info(f"⚙️ Created new {label} ID {settings.id}")
        return settings

    def _get_or_create_detection_settings(
        self, session: Session, detection_settings: DetectionSettings | int
    ) -> DetectionSettings:
        """Get existing or create new DetectionSettings in database."""
        from sqlalchemy.orm import exc as orm_exc

        from cali.sqlmodel._model import DetectionSettings

        # Handle detached objects that lost their session
        if not isinstance(detection_settings, int):
            try:
                _ = detection_settings.id
            except orm_exc.DetachedInstanceError:
                detection_settings = session.merge(detection_settings)
                cali_logger.info(
                    f"♻️ Reattached DetectionSettings ID {detection_settings.id}"  # type: ignore[union-attr]
                )
                return cast("DetectionSettings", detection_settings)

        return cast(
            "DetectionSettings",
            self._resolve_settings(
                session, detection_settings, DetectionSettings, "DetectionSettings"
            ),
        )

    def _get_or_create_extraction_settings(
        self, session: Session, extraction_settings: ExtractionSettings | int
    ) -> ExtractionSettings:
        """Get existing or create new ExtractionSettings in database."""
        from cali.sqlmodel._model import ExtractionSettings

        return cast(
            "ExtractionSettings",
            self._resolve_settings(
                session, extraction_settings, ExtractionSettings, "ExtractionSettings"
            ),
        )

    def _get_or_create_analysis_settings(
        self, session: Session, analysis_settings: AnalysisSettings | int
    ) -> AnalysisSettings:
        """Get existing or create new AnalysisSettings in database."""
        from cali.sqlmodel._model import AnalysisSettings

        return cast(
            "AnalysisSettings",
            self._resolve_settings(
                session,
                analysis_settings,
                AnalysisSettings,
                "AnalysisSettings",
                use_merge=True,
            ),
        )

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
        source_extraction_settings_id: int | None = None,
        source_detection_settings_id: int | None = None,
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
        source_extraction_settings_id : int | None
            When include_traces is False (analysis-only), the extraction settings
            ID to find the correct source traces to copy.
        source_detection_settings_id : int | None
            When include_traces is False (analysis-only), the detection settings
            ID to find the correct source traces to copy.
        """
        if analysis_result_id is None:
            return

        for roi in fov.rois:
            # Process traces
            if include_traces and hasattr(roi, "_new_traces"):
                # Extraction path: use newly created traces
                for trace in roi._new_traces:
                    trace.analysis_result_id = analysis_result_id
                    roi.traces_history.append(trace)
                delattr(roi, "_new_traces")
            elif not include_traces and roi.id is not None:
                # Analysis-only path: copy existing traces to new run
                source_trace = self._find_source_trace(
                    session,
                    roi.id,
                    source_extraction_settings_id,
                    source_detection_settings_id,
                )
                if source_trace is not None:
                    new_trace = Traces(
                        raw_trace=source_trace.raw_trace,
                        corrected_trace=source_trace.corrected_trace,
                        neuropil_trace=source_trace.neuropil_trace,
                        dff=source_trace.dff,
                        den_dff=source_trace.den_dff,
                        inferred_spikes=source_trace.inferred_spikes,
                        x_axis=source_trace.x_axis,
                        x_axis_units=source_trace.x_axis_units,
                        roi_id=roi.id,
                        analysis_result_id=analysis_result_id,
                        neuropil_mask_id=source_trace.neuropil_mask_id,
                    )
                    session.add(new_trace)

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

    @staticmethod
    def _find_source_trace(
        session: Session,
        roi_id: int,
        extraction_settings_id: int | None,
        detection_settings_id: int | None = None,
    ) -> Traces | None:
        """Find the source trace for an ROI from a specific extraction run.

        Parameters
        ----------
        session : Session
            Database session
        roi_id : int
            ROI ID to find traces for
        extraction_settings_id : int | None
            Extraction settings ID to match. If None, returns most recent trace.
        detection_settings_id : int | None
            Detection settings ID to match. Ensures we find traces from the
            correct detection+extraction combination.
        """
        from sqlmodel import col, select

        stmt = select(Traces).where(col(Traces.roi_id) == roi_id)
        if extraction_settings_id is not None or detection_settings_id is not None:
            stmt = stmt.join(CaliResult)
            if extraction_settings_id is not None:
                stmt = stmt.where(
                    col(CaliResult.extraction_settings_id) == extraction_settings_id
                )
            if detection_settings_id is not None:
                stmt = stmt.where(
                    col(CaliResult.detection_settings_id) == detection_settings_id
                )
        stmt = stmt.order_by(col(Traces.id).desc()).limit(1)
        return session.exec(stmt).first()  # type: ignore

    def _run_analysis_only(
        self,
        analysis_settings: AnalysisSettings,
        fovs: Iterable[FOV],
    ) -> Generator[FOV, None, None]:
        """Run analysis only on FOVs that already have extraction results.

        Uses the AnalysisRunner to compute both ROI-level analysis
        (DataAnalysis records) and FOV-level analysis (FOVAnalysis records).

        Parameters
        ----------
        analysis_settings : AnalysisSettings
            Analysis configuration (peak detection, thresholds)
        fovs : Iterable[FOV]
            FOVs with ROIs and existing traces to analyze
        """
        cali_logger.info("📊 Running Analysis (using existing extraction)...")
        self._analysis_runner = AnalysisRunner()
        yield from self._analysis_runner.run(
            fovs=fovs,
            analysis_settings=analysis_settings,
            as_generator=True,
        )

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
            msg = (
                "At least one of positions_detected, positions_extracted, "
                "or positions_analyzed must be provided"
            )
            cali_logger.error(msg)
            raise ValueError(msg)

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
            self._merge_result_positions(
                exact_match,
                positions_detected,
                positions_extracted,
                positions_analyzed,
            )
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
                self._merge_result_positions(
                    upgradeable_result,
                    positions_detected,
                    positions_extracted,
                    positions_analyzed,
                )
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
                self._merge_result_positions(
                    any_result_with_detection,
                    positions_detected,
                    None,
                    None,
                )
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
                self._merge_result_positions(
                    upgradeable_result,
                    positions_detected,
                    positions_extracted,
                    positions_analyzed,
                )
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
                self._merge_result_positions(
                    compatible_result,
                    positions_detected,
                    positions_extracted,
                    positions_analyzed,
                )
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

    @staticmethod
    def _merge_positions(
        existing: list[int] | None, incoming: list[int] | None
    ) -> list[int] | None:
        """Merge existing and incoming position lists, returning sorted union.

        Returns None if incoming is None (no update requested).
        """
        if incoming is None:
            return existing
        return sorted(set(existing or []) | set(incoming))

    def _merge_result_positions(
        self,
        result: CaliResult,
        positions_detected: list[int] | None,
        positions_extracted: list[int] | None,
        positions_analyzed: list[int] | None,
    ) -> None:
        """Merge position lists into an existing CaliResult."""
        result.positions_detected = self._merge_positions(
            result.positions_detected, positions_detected
        )
        result.positions_extracted = self._merge_positions(
            result.positions_extracted, positions_extracted
        )
        result.positions_analyzed = self._merge_positions(
            result.positions_analyzed, positions_analyzed
        )

    def _commit_fov_batch(
        self,
        fov: FOV,
        session: Session,
        experiment: Experiment,
        fov_count: int,
        total: int,
        detection_settings_id: int | None = None,
    ) -> bool:
        """Commit a single FOV result with batch logging.

        Commits to the database every `self.commit_batch_size` FOVs.

        Returns
        -------
        bool
            True if a batch commit was performed.
        """
        should_commit = fov_count % self.commit_batch_size == 0
        if should_commit:
            cali_logger.info(
                f"💾 Committing batch of {self.commit_batch_size} FOVs "
                f"(total: {fov_count}/{total})..."
            )
        commit_fov_result(
            session, experiment, fov, detection_settings_id, commit=should_commit
        )
        if should_commit:
            cali_logger.info(
                f"💾 Committed batch of {self.commit_batch_size} FOVs "
                f"(total: {fov_count}/{total})"
            )
        return should_commit

    def _commit_remaining_fovs(self, fov_count: int, session: Session) -> None:
        """Commit any remaining uncommitted FOVs in the current batch."""
        uncommitted = fov_count % self.commit_batch_size
        if uncommitted > 0:
            cali_logger.info(f"💾 Committing final batch of {uncommitted} FOVs...")
            session.commit()
            cali_logger.info(f"💾 Committed final batch of {uncommitted} FOVs.")

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
