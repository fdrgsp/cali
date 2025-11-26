"""Unified runner interface for detection and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import event
from sqlmodel import Session, create_engine, select

from cali._constants import DEFAULT_CALI_DB_NAME
from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.logger import cali_logger
from cali.sqlmodel import save_experiment_to_database
from cali.sqlmodel._model import FOV, ROI, CaliResult, Traces
from cali.util import commit_fov_result, load_data

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from cali.readers._ome_zarr_reader import OMEZarrReader
    from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, Experiment


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
        # database path
        self._db_path: Path | None = None
        self.commit_batch_size = commit_batch_size

        self._detection_runner = DetectionRunner()
        self._analysis_runner = AnalysisRunner()

    @property
    def database_path(self) -> str | None:
        """Get the database path."""
        return str(self._db_path) if self._db_path is not None else None

    def cancel(self) -> None:
        """Cancel both detection and analysis processes."""
        self._detection_runner.cancel()
        self._analysis_runner.cancel()

    def run(
        self,
        experiment: Experiment,
        dataset_path: str | Path,
        detection_settings: DetectionSettings | int,
        *,
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
        analysis_settings : AnalysisSettings | int | None
            Analysis parameters. It can be either an AnalysisSettings instance,
            an integer ID referencing an existing AnalysisSettings in the database,
            or None. If None, only detection is run. If provided, both detection and
            analysis are run.
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
        dataset = load_data(dataset_path)

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
        engine = create_engine(f"sqlite:///{self._db_path}", echo=echo)

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

                if analysis_settings is not None:
                    analysis_settings = self._get_or_create_analysis_settings(
                        session, analysis_settings
                    )

                assert detection_settings.id is not None

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
                    session,
                    detection_settings.id,
                    global_position_indices,
                )

                # 5. Run detection if needed
                fovs_with_rois = []
                positions_processed_detection = []
                total_rois_detected = 0
                if positions_for_detection:
                    yield "🔍 Running Detection..."
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

                        # Commit in batches
                        should_commit = fov_count % self.commit_batch_size == 0
                        commit_fov_result(
                            session,
                            experiment,
                            fov,
                            detection_settings.id,
                            commit=should_commit,
                        )
                        if should_commit:
                            cali_logger.info(
                                f"💾 Committed batch of {self.commit_batch_size} FOVs "
                                f"(total: {fov_count}/{len(positions_for_detection)})"
                            )

                        # Make FOV transient to keep it in memory after commit
                        # This allows us to use the threaded path for analysis
                        from sqlalchemy.orm import make_transient

                        # Save detection_settings_id before making transient
                        # (make_transient clears foreign keys)
                        detection_id = detection_settings.id

                        make_transient(fov)
                        for roi in fov.rois:
                            make_transient(roi)
                            # Restore detection_settings_id after make_transient
                            roi.detection_settings_id = detection_id
                            if roi.roi_mask:
                                make_transient(roi.roi_mask)

                        positions_processed_detection.append(fov.position_index)
                        fovs_with_rois.append(fov)

                    # Final commit for any remaining FOVs
                    if fov_count % self.commit_batch_size != 0:
                        session.commit()
                        remaining = fov_count % self.commit_batch_size
                        cali_logger.info(
                            f"💾 Final commit for remaining {remaining} FOVs"
                        )

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
                                detection_settings_id=detection_settings.id,
                                analysis_settings_id=None,
                                positions_analyzed=positions_processed_detection,
                            )

                # 7. Run analysis if settings provided
                if analysis_settings is not None:
                    assert analysis_settings.id is not None

                    yield "📊 Running Analysis..."

                    # Determine which positions need analysis
                    positions_for_analysis = self._get_positions_for_analysis(
                        session,
                        detection_settings.id,
                        analysis_settings.id,
                        global_position_indices,
                    )

                    if not positions_for_analysis:
                        return

                    # Create CaliResult FIRST with expected positions so we have the ID
                    analysis_result_id = None
                    if experiment.id is not None:
                        analysis_result_id = self._create_or_update_analysis_result(
                            session=session,
                            experiment_id=experiment.id,
                            detection_settings_id=detection_settings.id,
                            analysis_settings_id=analysis_settings.id,
                            positions_analyzed=positions_for_analysis,
                        )

                    # Load FOVs from database if not already in memory
                    if not fovs_with_rois or {
                        f.position_index for f in fovs_with_rois
                    } != set(positions_for_analysis):
                        cali_logger.info(
                            "📥 Loading FOVs from database for analysis..."
                        )
                        fovs_with_rois = self._load_fovs_from_db(
                            session,
                            detection_settings.id,
                            positions_for_analysis,
                        )

                    # Always use threaded processing
                    cali_logger.info(
                        f"⚡️ Running analysis with {analysis_settings.threads} threads"
                    )

                    # Run analysis and commit FOVs
                    positions_processed = []
                    fov_count = 0
                    for fov in self._run_analysis(
                        dataset,
                        analysis_settings,
                        detection_settings.id,
                        positions_for_analysis,
                        fovs_with_rois=fovs_with_rois,
                    ):
                        # Move new traces/analysis from temporary storage to actual collections
                        # and set analysis_result_id
                        if analysis_result_id is not None:
                            for roi in fov.rois:
                                # Process temporary new traces
                                if hasattr(roi, '_new_traces'):
                                    for trace in roi._new_traces:  # type: ignore
                                        trace.analysis_result_id = analysis_result_id
                                        roi.traces_history.append(trace)
                                    delattr(roi, '_new_traces')

                                # Process temporary new data analysis
                                if hasattr(roi, '_new_data_analysis'):
                                    for data_analysis in roi._new_data_analysis:  # type: ignore
                                        data_analysis.analysis_result_id = analysis_result_id
                                        roi.data_analysis_history.append(data_analysis)
                                    delattr(roi, '_new_data_analysis')

                        fov_count += 1
                        should_commit = fov_count % self.commit_batch_size == 0
                        commit_fov_result(
                            session, experiment, fov, commit=should_commit
                        )
                        if should_commit:
                            cali_logger.info(
                                f"💾 Committed batch of "
                                f"{self.commit_batch_size} FOVs "
                                f"(total: {fov_count}/"
                                f"{len(positions_for_analysis)})"
                            )
                        positions_processed.append(fov.position_index)

                    # Final commit for any remaining FOVs
                    if fov_count % self.commit_batch_size != 0:
                        session.commit()
                        remaining = fov_count % self.commit_batch_size
                        cali_logger.info(
                            f"💾 Final commit for remaining {remaining} FOVs"
                        )

                    # Log completion
                    if positions_processed:
                        cali_logger.info("✅ Analysis complete!")
                        cali_logger.info(
                            f"✅ Analysis committed: {fov_count} FOVs"
                        )
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
            engine = create_engine(f"sqlite:///{db_path}", echo=False)
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
        dataset: TensorstoreZarrReader | OMEZarrReader,
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
        )

    def _run_analysis(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader,
        analysis_settings: AnalysisSettings,
        detection_settings_id: int,
        global_position_indices: Sequence[int],
        fovs_with_rois: list[FOV],
    ) -> Generator[FOV, None, None]:
        """Run analysis using AnalysisRunner.

        Yields FOV results from AnalysisRunner using threaded processing.

        Parameters
        ----------
        dataset : TensorstoreZarrReader | OMEZarrReader
            Dataset reader
        analysis_settings : AnalysisSettings
            Analysis configuration
        detection_settings_id : int
            Detection settings ID
        global_position_indices : Sequence[int]
            Positions to analyze
        fovs_with_rois : list[FOV]
            FOVs with ROIs to analyze (from detection or loaded from DB)
        """
        cali_logger.info("📊 Running analysis...")
        yield from self._analysis_runner.run(
            dataset=dataset,
            settings=analysis_settings,
            fovs_with_rois=fovs_with_rois,
            global_position_indices=global_position_indices,
            detection_settings_id=detection_settings_id,
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

        fovs = session.exec(
            select(FOV)
            .where(FOV.position_index.in_(position_indices))  # type: ignore
            .options(
                joinedload(FOV.rois)  # type: ignore
                .joinedload(ROI.roi_mask),  # type: ignore
                joinedload(FOV.rois)  # type: ignore
                .joinedload(ROI.traces_history),  # type: ignore
                joinedload(FOV.rois)  # type: ignore
                .joinedload(ROI.data_analysis_history),  # type: ignore
            )
        ).unique().all()

        # Filter to only include ROIs with matching detection_settings_id
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
        analysis_settings_id: int | None,
        positions_analyzed: list[int],
    ) -> int:
        """Create or update an CaliResult entry.

        If a CaliResult with the same experiment, detection_settings, and
        analysis_settings already exists, update its positions_analyzed list
        by merging new positions. Otherwise create a new entry.

        Returns
        -------
            The ID of the created or updated CaliResult.
        """
        # Check if result with same settings already exists
        existing_result = session.exec(
            select(CaliResult).where(
                CaliResult.experiment == experiment_id,
                CaliResult.detection_settings == detection_settings_id,
                CaliResult.analysis_settings == analysis_settings_id,
            )
        ).first()

        if existing_result:
            # Update existing result by merging positions
            old_positions = set(existing_result.positions_analyzed or [])
            new_positions = set(positions_analyzed)
            merged_positions = sorted(old_positions | new_positions)

            existing_result.positions_analyzed = merged_positions
            session.add(existing_result)
            session.commit()
            session.refresh(existing_result)

            result_type = (
                "detection-only" if analysis_settings_id is None else "full analysis"
            )
            cali_logger.info(
                f"📝 Updated {result_type} CaliResult ID {existing_result.id} "
                f"(DetectionSettings={detection_settings_id}, "
                f"AnalysisSettings={analysis_settings_id}, "
                f"positions={merged_positions})"
            )
            assert existing_result.id is not None
            return existing_result.id
        else:
            # If creating a full analysis result, delete any detection-only result
            # with same detection settings (the analysis result supersedes it)
            if analysis_settings_id is not None:
                detection_only_result = session.exec(
                    select(CaliResult).where(
                        CaliResult.experiment == experiment_id,
                        CaliResult.detection_settings == detection_settings_id,
                        CaliResult.analysis_settings.is_(None),  # type: ignore
                    )
                ).first()

                if detection_only_result:
                    cali_logger.info(
                        "🗑️  Removing detection-only result ID "
                        f"{detection_only_result.id} (superseded by analysis result)"
                    )
                    session.delete(detection_only_result)
                    session.commit()

            # Create new result
            result = CaliResult(
                experiment=experiment_id,
                detection_settings=detection_settings_id,
                analysis_settings=analysis_settings_id,
                positions_analyzed=positions_analyzed,
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            result_type = (
                "detection-only" if analysis_settings_id is None else "full analysis"
            )
            cali_logger.info(
                f"📊 Created {result_type} CaliResult ID {result.id} "
                f"(DetectionSettings={detection_settings_id}, "
                f"AnalysisSettings={analysis_settings_id}, "
                f"positions={positions_analyzed})"
            )
            assert result.id is not None
            return result.id
