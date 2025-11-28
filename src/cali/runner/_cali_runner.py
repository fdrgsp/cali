"""Unified runner interface for detection and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import event
from sqlmodel import Session, create_engine, select

from cali._constants import DEFAULT_CALI_DB_NAME
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

    def __init__(self, commit_batch_size: int = 5) -> None:
        """Initialize the unified runner.

        Parameters
        ----------
        commit_batch_size : int
            Number of FOVs to accumulate before committing to database.
            Default is 5. Set to 1 for immediate commits (safest but slowest).
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
        echo: bool = False,
        as_generator: bool = False,
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
            Extraction parameters (neuropil, dff_window, decay_constant, etc.).
            Required if analysis_settings is provided. Can be an ExtractionSettings
            instance or integer ID referencing existing settings in the database.
        analysis_settings : AnalysisSettings | int | None
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
        echo : bool
            Enable SQLAlchemy echo for database operations
        as_generator : bool
            If True, returns a Generator that yields progress strings.
            If False (default), executes directly and silently consumes
            the generator.

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
            echo=echo,
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
        echo: bool = False,
    ) -> Generator[str, None, None]:
        """Internal generator for run progress.

        Yields progress strings during execution.
        """
        # 0. Make sure data are ready
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
        @event.listens_for(engine, "connect")
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

                if extraction_settings is not None:
                    extraction_settings = self._get_or_create_extraction_settings(
                        session, extraction_settings
                    )
                    extraction_settings_id = extraction_settings.id
                    extraction_threads = extraction_settings.threads
                    assert extraction_settings_id is not None
                else:
                    extraction_settings_id = None
                    extraction_threads = None

                if analysis_settings is not None:
                    analysis_settings = self._get_or_create_analysis_settings(
                        session, analysis_settings
                    )
                    analysis_settings_id = analysis_settings.id
                    assert analysis_settings_id is not None
                else:
                    analysis_settings_id = None

                det_id = detection_settings.id
                assert det_id is not None

                # 4. Determine which positions need detection
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
                    session, det_id, global_position_indices
                )

                yield "🔍 Running Detection..."

                # 5. Run detection if needed
                positions_processed_detection = []
                total_rois_detected = 0
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

                        # Commit in batches
                        should_commit = fov_count % self.commit_batch_size == 0
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
                            # Clear session to free memory
                            session.expunge_all()
                            # Re-attach settings objects for next iteration
                            detection_settings = session.merge(detection_settings)
                            if extraction_settings is not None:
                                extraction_settings = session.merge(extraction_settings)
                            if analysis_settings is not None:
                                analysis_settings = session.merge(analysis_settings)

                        positions_processed_detection.append(fov.position_index)

                    # Final commit for any remaining FOVs
                    if fov_count % self.commit_batch_size != 0:
                        session.commit()
                        remaining = fov_count % self.commit_batch_size
                        cali_logger.info(
                            f"💾 Final commit for remaining {remaining} FOVs"
                        )
                        session.expunge_all()
                        # Re-attach settings objects
                        detection_settings = session.merge(detection_settings)
                        if extraction_settings is not None:
                            extraction_settings = session.merge(extraction_settings)
                        if analysis_settings is not None:
                            analysis_settings = session.merge(analysis_settings)

                    # Log detection completion
                    if positions_processed_detection:
                        cali_logger.info(
                            f"✅ Detection committed: {total_rois_detected} "
                            f"ROIs across {len(positions_processed_detection)} FOVs"
                        )
                        # Only create detection-only result if no analysis will follow
                        if analysis_settings is None and experiment.id is not None:
                            self._create_or_update_analysis_result(
                                session=session,
                                experiment_id=experiment.id,
                                detection_settings_id=det_id,
                                extraction_settings_id=None,
                                analysis_settings_id=None,
                                positions_analyzed=positions_processed_detection,
                            )

                # 7. Run extraction if settings provided
                if extraction_settings is not None:
                    # Eagerly load all scalar attributes before detaching
                    # This prevents DetachedInstanceError when accessing later
                    _ = (
                        extraction_settings.threads,
                        extraction_settings.neuropil_inner_radius,
                        extraction_settings.neuropil_min_pixels,
                        extraction_settings.neuropil_correction_factor,
                        extraction_settings.decay_constant,
                        extraction_settings.dff_window,
                    )

                    # Detach settings for thread safety
                    # This prevents "session is in committed state" errors in threads
                    session.expunge(extraction_settings)

                    if analysis_settings is not None:
                        # Eagerly load all scalar attributes before detaching
                        _ = (
                            analysis_settings.threads,
                            analysis_settings.peaks_height_value,
                            analysis_settings.peaks_height_mode,
                            analysis_settings.peaks_distance,
                            analysis_settings.peaks_prominence_multiplier,
                            analysis_settings.calcium_sync_jitter_window,
                            analysis_settings.calcium_network_threshold,
                            analysis_settings.spike_threshold_value,
                            analysis_settings.spike_threshold_mode,
                            analysis_settings.burst_threshold,
                            analysis_settings.burst_min_duration,
                            analysis_settings.burst_gaussian_sigma,
                            analysis_settings.spikes_sync_cross_corr_lag,
                            analysis_settings.experiment_type,
                            analysis_settings.stimulation_mask_path,
                            analysis_settings.led_power_equation,
                            analysis_settings.led_pulse_duration,
                            analysis_settings.led_pulse_powers,
                            analysis_settings.led_pulse_on_frames,
                        )

                        # Load stimulation mask from file if path provided but
                        # mask not yet loaded
                        if (
                            analysis_settings.stimulation_mask_path
                            and analysis_settings.stimulation_mask is None
                        ):
                            import tifffile

                            from cali.sqlmodel._model import Mask
                            from cali.util import mask_to_coordinates

                            stim_mask_file = Path(
                                analysis_settings.stimulation_mask_path
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
                                analysis_settings.stimulation_mask = stimulation_mask
                                analysis_settings.stimulation_mask_id = (
                                    stimulation_mask.id
                                )
                                session.add(analysis_settings)
                                session.commit()
                                cali_logger.info(
                                    f"🎭 Loaded stimulation mask from "
                                    f"{stim_mask_file.name}"
                                )

                        # Ensure stimulation_mask is detached for thread safety
                        if analysis_settings.stimulation_mask:
                            session.expunge(analysis_settings.stimulation_mask)
                        session.expunge(analysis_settings)

                    yield "📈 Running Extraction" + (
                        " and Analysis..." if analysis_settings else "..."
                    )

                    # Determine which positions need extraction/analysis
                    # Use analysis_settings_id if doing analysis,
                    # otherwise None for extraction-only
                    analysis_id_for_check = analysis_settings_id

                    positions_for_extraction = (
                        self._get_positions_for_analysis(
                            session,
                            det_id,
                            analysis_id_for_check,
                            global_position_indices,
                        )
                        if analysis_id_for_check is not None
                        else global_position_indices
                    )

                    if not positions_for_extraction:
                        return

                    yield f"PROGRESS:RESET:{len(positions_for_extraction)}"

                    # Create CaliResult FIRST with expected positions so we have the ID
                    analysis_result_id = None
                    if experiment.id is not None:
                        analysis_result_id = self._create_or_update_analysis_result(
                            session=session,
                            experiment_id=experiment.id,
                            detection_settings_id=det_id,
                            extraction_settings_id=extraction_settings_id,
                            analysis_settings_id=analysis_id_for_check,
                            positions_analyzed=list(positions_for_extraction),
                        )

                    # Process in batches
                    # Use a batch size that is at least the number of threads to
                    # ensure utilization but not too large to consume too much memory.
                    # Default to commit_batch_size, but ensure min of threads
                    assert extraction_threads is not None
                    batch_size = max(self.commit_batch_size, extraction_threads)

                    positions_processed = []
                    fov_count = 0

                    for i in range(0, len(positions_for_extraction), batch_size):
                        batch_positions = positions_for_extraction[i : i + batch_size]

                        # Prepare FOVs for this batch
                        batch_fovs = []

                        cali_logger.info(
                            f"💿 Loading {len(batch_positions)} FOVs from database..."
                        )
                        loaded_fovs = self._load_fovs_from_db(
                            session, det_id, batch_positions
                        )

                        # Detach FOVs from session to allow safe threading
                        # We must do this because SQLAlchemy objects are not thread-safe
                        # and _run_analysis uses a ThreadPoolExecutor.
                        # Since we used selectinload in _load_fovs_from_db, all needed
                        # data (ROIs, masks, traces) is already loaded.
                        for fov in loaded_fovs:
                            session.expunge(fov)

                        batch_fovs.extend(loaded_fovs)

                        if not batch_fovs:
                            continue

                        # Run extraction on this batch
                        for fov in self._run_extraction(
                            dataset,
                            extraction_settings,
                            analysis_settings,
                            fovs=batch_fovs,
                        ):
                            # Move new traces/analysis from temporary storage to actual
                            # collections and set analysis_result_id
                            if analysis_result_id is not None:
                                for roi in fov.rois:
                                    # Process temporary new traces
                                    if hasattr(roi, "_new_traces"):
                                        for trace in roi._new_traces:  # type: ignore
                                            trace.analysis_result_id = (
                                                analysis_result_id
                                            )
                                            roi.traces_history.append(trace)
                                        delattr(roi, "_new_traces")

                                    # Process temporary new data analysis
                                    if hasattr(roi, "_new_data_analysis"):
                                        for data_analysis in roi._new_data_analysis:  # type: ignore
                                            data_analysis.analysis_result_id = (
                                                analysis_result_id
                                            )
                                            roi.data_analysis_history.append(
                                                data_analysis
                                            )
                                        delattr(roi, "_new_data_analysis")

                            fov_count += 1
                            yield "PROGRESS:UPDATE"
                            should_commit = fov_count % self.commit_batch_size == 0
                            commit_fov_result(
                                session, experiment, fov, commit=should_commit
                            )
                            if should_commit:
                                cali_logger.info(
                                    f"💾 Committed batch of "
                                    f"{self.commit_batch_size} FOVs "
                                    f"(total: {fov_count}/"
                                    f"{len(positions_for_extraction)})"
                                )
                            positions_processed.append(fov.position_index)

                        # Commit any remaining in this batch and clear memory
                        session.commit()
                        for fov in batch_fovs:
                            # Expunge to free memory
                            try:
                                session.expunge(fov)
                            except Exception:
                                pass

                    # Log completion
                    if positions_processed:
                        cali_logger.info(f"✅ Extraction committed: {fov_count} FOVs")
        finally:
            engine.dispose(close=True)

    # ==================== PRIVATE HELPER METHODS ====================

    def _get_positions_for_detection(
        self,
        session: Session,
        detection_settings_id: int,
        global_position_indices: Sequence[int],
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

        Returns
        -------
        list[int]
            Positions that need detection. Empty list means skip all.
        """
        # Check which positions already have ROIs with this detection
        existing_positions = session.exec(
            select(FOV.position_index)
            .join(ROI)
            .where(
                ROI.detection_settings_id == detection_settings_id,
                FOV.position_index.in_(global_position_indices),  # type: ignore
            )
            .distinct()
        ).all()

        existing_pos_set = set(existing_positions)
        positions_needing_detection = [
            p for p in global_position_indices if p not in existing_pos_set
        ]

        if not positions_needing_detection:
            cali_logger.info(
                "✓ Detection already exists for all positions with "
                f"DetectionSettings ID {detection_settings_id}. Skipping detection."
            )
            return []

        if existing_pos_set:
            cali_logger.info(
                f"⚠️  Detection exists for {len(existing_pos_set)} position(s) "
                f"but missing for {len(positions_needing_detection)} position(s): "
                f"{positions_needing_detection}. "
                "Running detection for missing positions."
            )

        return positions_needing_detection

    def _get_positions_for_analysis(
        self,
        session: Session,
        detection_settings_id: int,
        analysis_settings_id: int,
        global_position_indices: Sequence[int],
    ) -> list[int]:
        """Get positions that need analysis.

        Returns positions that either:
        - Don't have analysis results yet, OR
        - force=True (re-run all)

        Parameters
        ----------
        session : Session
            Database session
        detection_settings_id : int
            Detection settings ID to check
        analysis_settings_id : int
            Analysis settings ID to check
        global_position_indices : Sequence[int]
            Positions to check

        Returns
        -------
        list[int]
            Positions that need analysis. Empty list means skip all.
        """
        # Find positions that already have Traces with this analysis_settings_id
        # and detection_settings_id combination
        existing_positions = session.exec(
            select(FOV.position_index)
            .join(ROI)
            .join(Traces)
            .where(
                ROI.detection_settings_id == detection_settings_id,
                Traces.analysis_result_id.in_(  # type: ignore
                    select(CaliResult.id).where(
                        CaliResult.analysis_settings == analysis_settings_id
                    )
                ),
                FOV.position_index.in_(global_position_indices),  # type: ignore
            )
            .distinct()
        ).all()

        existing_pos_set = set(existing_positions)
        positions_needing_analysis = [
            p for p in global_position_indices if p not in existing_pos_set
        ]

        if not positions_needing_analysis:
            cali_logger.info(
                "ℹ️ Analysis already exists for all positions with DetectionSettings ID "
                f"{detection_settings_id} and AnalysisSettings ID "
                f"{analysis_settings_id}. Skipping analysis."
            )
            return []

        if existing_pos_set:
            cali_logger.info(
                f"ℹ️ Analysis exists for {len(existing_pos_set)} position(s) "
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
            cali_logger.info(
                f"♻️ Reusing existing DetectionSettings ID {existing.id} "
                f"(method: {existing.method})"
            )
            return existing

        elif detection_settings.id is None:
            # Check if identical settings already exist
            all_settings = session.exec(select(DetectionSettings)).all()
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
            cali_logger.info(f"♻️ Reusing existing ExtractionSettings ID {existing.id}")
            return existing

        elif extraction_settings.id is None:
            # Check if identical settings already exist
            all_settings = session.exec(select(ExtractionSettings)).all()
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
            cali_logger.info(f"♻️ Reusing existing AnalysisSettings ID {existing.id}")
            return existing

        elif analysis_settings.id is None:
            # Check if identical settings already exist
            all_settings = session.exec(select(AnalysisSettings)).all()
            for candidate in all_settings:
                if analysis_settings == candidate:
                    cali_logger.info(
                        f"♻️ Reusing existing AnalysisSettings ID {candidate.id}"
                    )
                    return candidate

            # New settings - merge and commit
            analysis_settings = session.merge(analysis_settings)
            session.commit()
            session.refresh(analysis_settings)
            cali_logger.info(
                f"⚙️ Created new AnalysisSettings ID {analysis_settings.id}"
            )
            return analysis_settings
        else:
            # Settings has ID - merge to reattach
            analysis_settings = session.merge(analysis_settings)
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
        cali_logger.info("🔍 Running detection...")
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
        cali_logger.info("📈 Running extraction...")
        yield from self._extraction_runner.run(
            dataset=dataset,
            extraction_settings=extraction_settings,
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

        fovs = (
            session.exec(
                select(FOV)
                .where(FOV.position_index.in_(position_indices))  # type: ignore
                .options(
                    joinedload(FOV.rois).joinedload(ROI.roi_mask),  # type: ignore  # type: ignore
                    joinedload(FOV.rois).joinedload(ROI.traces_history),  # type: ignore  # type: ignore
                    joinedload(FOV.rois).joinedload(ROI.data_analysis_history),  # type: ignore  # type: ignore
                )
            )
            .unique()
            .all()
        )

        # Filter to only include ROIs with matching detection_settings_id
        # Note: We can't easily filter eager loaded collections in the query
        # with selectinload so we filter in Python.
        filtered_fovs = []
        for fov in fovs:
            # Keep only ROIs with matching detection settings
            matching_rois = [
                roi
                for roi in fov.rois
                if roi.detection_settings_id == detection_settings_id
            ]
            if matching_rois:
                fov.rois = matching_rois
                filtered_fovs.append(fov)

        return filtered_fovs

    def _create_or_update_analysis_result(
        self,
        session: Session,
        experiment_id: int,
        detection_settings_id: int,
        extraction_settings_id: int | None,
        analysis_settings_id: int | None,
        positions_analyzed: list[int],
    ) -> int:
        """Create or update a CaliResult entry.

        Handles progressive analysis stages:
        1. If exact match exists (same detection, extraction, analysis), merge positions
        2. If partial match exists (same detection/extraction but less complete),
           upgrade the existing result with new settings
        3. Otherwise, create new result and clean up superseded ones

        Returns
        -------
            The ID of the created or updated CaliResult.
        """
        # First, check for exact match with all settings
        query = select(CaliResult).where(
            CaliResult.experiment == experiment_id,
            CaliResult.detection_settings == detection_settings_id,
        )

        if extraction_settings_id is None:
            query = query.where(CaliResult.extraction_settings.is_(None))  # type: ignore
        else:
            query = query.where(
                CaliResult.extraction_settings == extraction_settings_id
            )

        if analysis_settings_id is None:
            query = query.where(CaliResult.analysis_settings.is_(None))  # type: ignore
        else:
            query = query.where(CaliResult.analysis_settings == analysis_settings_id)

        exact_match = session.exec(query).first()

        if exact_match:
            # Update existing result by merging positions
            old_positions = set(exact_match.positions_analyzed or [])
            new_positions = set(positions_analyzed)
            merged_positions = sorted(old_positions | new_positions)

            exact_match.positions_analyzed = merged_positions
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
                f"positions={merged_positions})"
            )
            assert exact_match.id is not None
            return exact_match.id

        # Check for a less-complete result that can be upgraded
        # (same detection/extraction but missing analysis)
        if analysis_settings_id is not None and extraction_settings_id is not None:
            upgradeable_result = session.exec(
                select(CaliResult).where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings == detection_settings_id,
                    CaliResult.extraction_settings == extraction_settings_id,
                    CaliResult.analysis_settings.is_(None),  # type: ignore
                )
            ).first()

            if upgradeable_result:
                # Upgrade the existing result with analysis settings
                old_positions = set(upgradeable_result.positions_analyzed or [])
                new_positions = set(positions_analyzed)
                merged_positions = sorted(old_positions | new_positions)

                upgradeable_result.analysis_settings = analysis_settings_id
                upgradeable_result.positions_analyzed = merged_positions
                session.add(upgradeable_result)
                session.commit()
                session.refresh(upgradeable_result)

                cali_logger.info(
                    f"⬆️  Upgraded CaliResult ID {upgradeable_result.id} "
                    f"with AnalysisSettings={analysis_settings_id}, "
                    f"positions={merged_positions}"
                )
                assert upgradeable_result.id is not None
                return upgradeable_result.id

        # Check for result with same detection/extraction but missing extraction
        # (upgrade detection-only to detection+extraction)
        if extraction_settings_id is not None and analysis_settings_id is None:
            upgradeable_result = session.exec(
                select(CaliResult).where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings == detection_settings_id,
                    CaliResult.extraction_settings.is_(None),  # type: ignore
                    CaliResult.analysis_settings.is_(None),  # type: ignore
                )
            ).first()

            if upgradeable_result:
                # Upgrade the existing result with extraction settings
                old_positions = set(upgradeable_result.positions_analyzed or [])
                new_positions = set(positions_analyzed)
                merged_positions = sorted(old_positions | new_positions)

                upgradeable_result.extraction_settings = extraction_settings_id
                upgradeable_result.positions_analyzed = merged_positions
                session.add(upgradeable_result)
                session.commit()
                session.refresh(upgradeable_result)

                cali_logger.info(
                    f"⬆️  Upgraded CaliResult ID {upgradeable_result.id} "
                    f"with ExtractionSettings={extraction_settings_id}, "
                    f"positions={merged_positions}"
                )
                assert upgradeable_result.id is not None
                return upgradeable_result.id

        # No upgradeable result found - create new one
        # First, delete any detection-only result that will be superseded
        if analysis_settings_id is not None:
            detection_only_result = session.exec(
                select(CaliResult).where(
                    CaliResult.experiment == experiment_id,
                    CaliResult.detection_settings == detection_settings_id,
                    CaliResult.extraction_settings.is_(None),  # type: ignore
                    CaliResult.analysis_settings.is_(None),  # type: ignore
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
            detection_settings=detection_settings_id,
            extraction_settings=extraction_settings_id,
            analysis_settings=analysis_settings_id,
            positions_analyzed=positions_analyzed,
        )
        session.add(result)
        session.commit()
        session.refresh(result)

        result_type = self._get_result_type(
            extraction_settings_id, analysis_settings_id
        )
        cali_logger.info(
            f"📊 Created {result_type} CaliResult ID {result.id} "
            f"(DetectionSettings={detection_settings_id}, "
            f"ExtractionSettings={extraction_settings_id}, "
            f"AnalysisSettings={analysis_settings_id}, "
            f"positions={positions_analyzed})"
        )
        assert result.id is not None
        return result.id

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
