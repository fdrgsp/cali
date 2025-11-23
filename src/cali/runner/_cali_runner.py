"""Unified runner interface for detection and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from sqlmodel import Session, create_engine, select

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
    """

    def __init__(self) -> None:
        """Initialize the unified runner."""
        # database path
        self._db_path: Path | None = None

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
        output_path: Path | None = None,
        overwrite: bool = False,
        echo: bool = False,
    ) -> None:
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
        8. Creates AnalysisResult entry (always new, never overwrites)

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
        output_path : Path | None
            Output path to save databse and analysis results. If None, uses dataset
            parent directory.
        overwrite : bool
            Whether to overwrite existing database
        echo : bool
            Enable SQLAlchemy echo for database operations

        Raises
        ------
        ValueError
            If no FOVs exist for the specified detection settings and positions
        """
        # 0. Make sure data are ready
        dataset = load_data(dataset_path)

        if output_path is None:
            output_path = Path(dataset_path).parent

        # 1. Setup database
        if database_name is not None:
            db_name = (
                database_name
                if database_name.endswith(".cali")
                else f"{database_name}.cali"
            )
            self._db_path = output_path / db_name
        else:
            self._db_path = output_path / "results.cali"
        self._setup_database(self._db_path, experiment, overwrite)

        # 2. Get database engine and session
        engine = create_engine(f"sqlite:///{self._db_path}", echo=echo)
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
                    for fov in self._run_detection(
                        dataset,
                        detection_settings,
                        positions_for_detection,
                    ):
                        # Count ROIs before commit (they may become detached after)
                        roi_count = len(fov.rois)
                        total_rois_detected += roi_count

                        # Commit each FOV immediately to reduce memory usage
                        commit_fov_result(
                            session, experiment, fov, detection_settings.id
                        )
                        positions_processed_detection.append(fov.position_index)
                        fovs_with_rois.append(fov)

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

                    # Determine which positions need analysis
                    positions_for_analysis = self._get_positions_for_analysis(
                        session,
                        detection_settings.id,
                        analysis_settings.id,
                        global_position_indices,
                    )

                    if not positions_for_analysis:
                        return

                    # Load FOVs from database (only for positions needing analysis)
                    # After committing detection, the in-memory FOVs become detached
                    fovs_with_rois = self._load_fovs_from_database(
                        session,
                        detection_settings.id,
                        positions_for_analysis,
                    )

                    if not fovs_with_rois:
                        cali_logger.error(
                            "No FOVs with ROIs found. Run detection first."
                        )
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

                    # Now run analysis and commit FOVs one by one, setting
                    # analysis_result_id on Traces before committing
                    positions_processed = []
                    for fov in self._run_analysis(
                        dataset,
                        analysis_settings,
                        fovs_with_rois,
                        global_position_indices,
                    ):
                        # Set analysis_result_id on all Traces before committing this FOV
                        if analysis_result_id is not None:
                            for roi in fov.rois:
                                traces = (
                                    getattr(roi, "_new_traces", [])
                                    or roi.traces_history
                                )
                                for trace in traces:
                                    trace.analysis_result_id = analysis_result_id

                        # Commit immediately while temp lists are still attached
                        commit_fov_result(session, experiment, fov)
                        positions_processed.append(fov.position_index)

                    # Log completion
                    if positions_processed:
                        total_rois = sum(len(fov.rois) for fov in fovs_with_rois)
                        cali_logger.info(
                            f"✅ Analysis committed: {total_rois} "
                            f"ROIs across {len(positions_processed)} FOVs"
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
                f"{positions_needing_detection}. Running detection for missing positions."
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
                "✓ Analysis already exists for all positions with DetectionSettings ID "
                f"{detection_settings_id} and AnalysisSettings ID "
                f"{analysis_settings_id}. Skipping analysis."
            )
            return []

        if existing_pos_set:
            cali_logger.info(
                f"⚠️  Analysis exists for {len(existing_pos_set)} position(s) "
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
                    # Check if they match using __eq__ (compares name + type)
                    if experiment != db_exp:
                        msg = (
                            f"The provided Experiment (name='{experiment.name}', "
                            f"type='{experiment.experiment_type}') does not match the "
                            f"one in the database (name='{db_exp.name}', "
                            f"type='{db_exp.experiment_type}'). "
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
        fovs_with_rois: list[FOV],
        global_position_indices: Sequence[int],
    ) -> Generator[FOV, None, None]:
        """Run analysis using AnalysisRunner (pure computation).

        Yields FOV results from AnalysisRunner.
        """
        cali_logger.info("📊 Running analysis...")
        yield from self._analysis_runner.run(
            dataset=dataset,
            settings=analysis_settings,
            fovs_with_rois=fovs_with_rois,
            global_position_indices=global_position_indices,
        )

    def _create_or_update_analysis_result(
        self,
        session: Session,
        experiment_id: int,
        detection_settings_id: int,
        analysis_settings_id: int | None,
        positions_analyzed: list[int],
    ) -> int:
        """Create or update an AnalysisResult entry.

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
                f"📝 Updated {result_type} AnalysisResult ID {existing_result.id} "
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
                f"📊 Created {result_type} AnalysisResult ID {result.id} "
                f"(DetectionSettings={detection_settings_id}, "
                f"AnalysisSettings={analysis_settings_id}, "
                f"positions={positions_analyzed})"
            )
            assert result.id is not None
            return result.id

    def _cascade_delete_analysis_results(
        self,
        session: Session,
        experiment: Experiment,
        detection_settings_id: int,
    ) -> None:
        """Delete all AnalysisResults using these DetectionSettings.

        This cascades to delete Traces, DataAnalysis, and other related data.
        ROIs will be replaced during the new detection run.
        """
        # Find all AnalysisResults using these detection settings
        results_to_delete = session.exec(
            select(CaliResult).where(
                CaliResult.experiment == experiment.id,
                CaliResult.detection_settings == detection_settings_id,
            )
        ).all()

        if results_to_delete:
            count = len(results_to_delete)
            result_ids = [r.id for r in results_to_delete]
            cali_logger.warning(
                f"🗑️  force=True: Deleting {count} AnalysisResult(s) "
                f"(IDs: {result_ids}) and associated analysis data "
                f"for DetectionSettings ID {detection_settings_id}"
            )

            for result in results_to_delete:
                session.delete(result)
            session.commit()

    def _load_fovs_from_database(
        self,
        session: Session,
        detection_settings_id: int,
        global_position_indices: Sequence[int],
    ) -> list[FOV]:
        """Load FOVs with ROIs from database for analysis.

        Used when skip_detection=True to load existing ROIs.
        Only loads ROIs matching the specified detection_settings_id.
        """
        from sqlalchemy.orm import selectinload

        fovs = []
        for pos_idx in global_position_indices:
            # Query for FOV at this position that has ROIs with
            # matching detection_settings_id
            fov_stmt = (
                select(FOV)
                .join(ROI)
                .where(
                    FOV.position_index == pos_idx,
                    ROI.detection_settings_id == detection_settings_id,
                )
                .options(
                    selectinload(FOV.rois).selectinload(ROI.roi_mask),
                )
            )
            fov = session.exec(fov_stmt).first()

            if fov is None:
                cali_logger.warning(
                    f"No FOV found at position {pos_idx} with "
                    f"detection_settings_id={detection_settings_id}. Skipping."
                )
                continue

            # Filter ROIs to only include those matching detection_settings_id
            # (The query loads the FOV, but relationships load all ROIs)
            fov.rois = [
                roi
                for roi in fov.rois
                if roi.detection_settings_id == detection_settings_id
            ]

            if not fov.rois:
                cali_logger.warning(
                    f"FOV at position {pos_idx} has no ROIs with "
                    f"detection_settings_id={detection_settings_id}. Skipping."
                )
                continue

            fovs.append(fov)

        if not fovs:
            cali_logger.error(
                f"No FOVs found for detection_settings_id="
                f"{detection_settings_id} at positions {list(global_position_indices)}"
            )

        return fovs
