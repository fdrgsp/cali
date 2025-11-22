"""Detection runner for executing segmentation and saving to database."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from sqlmodel import Session, create_engine, select
from tqdm import tqdm

from cali._constants import EVENT_KEY
from cali.sqlmodel._model import FOV, ROI, DetectionSettings, Experiment, Mask
from cali.sqlmodel._util import save_experiment_to_database
from cali.util import commit_fov_result, load_data, mask_to_coordinates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellpose.models import CellposeModel

    from cali.readers import OMEZarrReader, TensorstoreZarrReader

from cali.logger import cali_logger


def _get_fov_name(event_key: str, meta: list[dict], global_pos_idx: int) -> str:
    """Get the FOV name from metadata."""
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
        return f"{well}_{global_pos_idx:04d}"
    except (KeyError, IndexError):
        pass

    # Final fallback
    return f"pos_{global_pos_idx}"


class DetectionRunner:
    """Runner for neuron detection that saves masks directly to database.

    Similar to AnalysisRunner but for the detection/segmentation phase.
    Supports both Cellpose and CaImAn detection methods.
    Creates FOV and ROI objects with masks in the database.
    """

    def __init__(self) -> None:
        super().__init__()
        # The data reader
        self._data: TensorstoreZarrReader | OMEZarrReader | None = None
        # Use threading.Event for cancellation control
        self._cancellation_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the detection process."""
        self._cancellation_event.set()
        cali_logger.info("Cancellation requested...")

    def run(
        self,
        experiment: Experiment,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
        overwrite: bool = False,
        force: bool = False,
        echo: bool = False,
    ) -> None:
        """Run detection and save masks to database.

        Automatically selects the appropriate detection method based on
        detection_settings.method ("cellpose" or "caiman").

        Parameters
        ----------
        experiment : Experiment
            Experiment to add detection results to
        detection_settings : DetectionSettings
            Detection parameters (method field determines which algorithm to use)
        global_position_indices : Sequence[int]
            Position indices to process
        overwrite : bool
            Whether to overwrite existing database
        force : bool
            If True, delete all existing analysis results using these detection
            settings and re-run detection. If False (default), skip detection if
            settings+positions are unchanged, or run only for new positions.
        echo : bool
            Enable SQLAlchemy echo for database operations
        """
        if detection_settings.method == "cellpose":
            self._run_cellpose(
                experiment=experiment,
                detection_settings=detection_settings,
                global_position_indices=global_position_indices,
                overwrite=overwrite,
                force=force,
                echo=echo,
            )
        elif detection_settings.method == "caiman":
            self._run_caiman(
                experiment=experiment,
                detection_settings=detection_settings,
                global_position_indices=global_position_indices,
                overwrite=overwrite,
                force=force,
                echo=echo,
            )
        else:
            msg = (
                f"Unknown detection method: {detection_settings.method}. "
                "Supported methods: 'cellpose', 'caiman'"
            )
            cali_logger.error(msg)
            raise ValueError(msg)

    def _run_caiman(
        self,
        experiment: Experiment,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
        overwrite: bool = False,
        force: bool = False,
        echo: bool = False,
    ) -> None:
        """Run CaImAn detection and save masks to database.

        Parameters
        ----------
        experiment : Experiment
            Experiment to add detection results to
        detection_settings : DetectionSettings
            Detection parameters (method should be "caiman")
        global_position_indices : Sequence[int]
            Position indices to process
        overwrite : bool
            Whether to overwrite existing database
        force : bool
            If True, delete existing analysis and re-run. If False, skip if unchanged
            or run only new positions.
        echo : bool
            Enable SQLAlchemy echo for database operations
        """
        # TODO: Implement CaImAn detection
        cali_logger.warning("CaImAn detection not yet implemented")
        raise NotImplementedError("CaImAn detection coming soon!")

    def _run_cellpose(
        self,
        experiment: Experiment,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
        overwrite: bool = False,
        force: bool = False,
        cellpose_debug: bool = False,
        echo: bool = False,
    ) -> None:
        """Run Cellpose segmentation and save masks to database.

        Parameters
        ----------
        experiment : Experiment
            Experiment to add detection results to
        detection_settings : DetectionSettings
            Detection parameters (method should be "cellpose")
        global_position_indices : Sequence[int]
            Position indices to process
        overwrite : bool
            Whether to overwrite existing database
        force : bool
            If True, delete existing analysis and re-run. If False, skip if unchanged
            or run only new positions.
        cellpose_debug : bool
            Enable Cellpose debug logging
        echo : bool
            Enable SQLAlchemy echo for database operations
        """
        try:
            from cellpose import core, io
            from cellpose.models import CellposeModel
        except ImportError as e:
            cali_logger.error(
                "Cellpose is not installed. Please install Cellpose to use "
                "Cellpose detection: `uv sync --extra cp4`."
            )
            raise e

        if cellpose_debug:
            io.logger_setup()

        use_gpu = core.use_gpu()
        cali_logger.info(f"Use GPU: {use_gpu}")

        cali_logger.info(f"Loading model from `{detection_settings.model_type}`.")
        model = CellposeModel(
            pretrained_model=str(detection_settings.model_type), gpu=use_gpu
        )

        # Run detection and get FOV results
        fov_results = self._run_cellpose_detection(
            experiment=experiment,
            global_position_indices=global_position_indices,
            model=model,
            diameter=detection_settings.diameter,
            cellprob_threshold=detection_settings.cellprob_threshold,
            flow_threshold=detection_settings.flow_threshold,
            batch_size=detection_settings.batch_size,
            min_size=detection_settings.min_size,
            normalize=detection_settings.normalize,
            overwrite=overwrite,
        )

        # Commit results to database
        if fov_results:
            cali_logger.info(
                f"ROIs detected: {sum(len(fov.rois) for fov in fov_results)}"
            )
            cali_logger.info("Committing detection results to database...")
            engine = create_engine(f"sqlite:///{experiment.db_path}", echo=echo)
            with Session(engine) as session:
                # Check if identical detection settings already exist
                detection_settings = self._get_or_create_detection_settings(
                    session, detection_settings
                )

                # Determine which positions to actually save based on force flag
                positions_to_save = []
                for fov_result in fov_results:
                    positions_to_save.append(fov_result.position_index)

                if not force:
                    # Check which positions already have ROIs
                    existing_positions = session.exec(
                        select(FOV.position_index)
                        .join(ROI)
                        .where(
                            ROI.detection_settings_id == detection_settings.id,
                            FOV.position_index.in_(positions_to_save),  # type: ignore
                        )
                        .distinct()
                    ).all()

                    existing_pos_set = set(existing_positions)
                    new_positions = [
                        p for p in positions_to_save if p not in existing_pos_set
                    ]

                    if not new_positions and existing_pos_set:
                        # All positions already exist - skip
                        cali_logger.info(
                            f"⏭️  Skipping detection - all {len(positions_to_save)} "
                            f"position(s) already have ROIs with DetectionSettings ID "
                            f"{detection_settings.id}. Use force=True to re-run and "
                            f"replace existing analysis."
                        )
                        engine.dispose(close=True)
                        return

                    if new_positions and len(new_positions) < len(positions_to_save):
                        # Some positions already exist - only save new ones
                        cali_logger.info(
                            f"📍 Saving detection for {len(new_positions)} new "
                            f"position(s): {new_positions} (skipping "
                            f"{len(existing_pos_set)} existing)"
                        )
                        # Filter fov_results to only new positions
                        fov_results = [
                            fov
                            for fov in fov_results
                            if fov.position_index in new_positions
                        ]
                        positions_to_save = new_positions

                if force and detection_settings.id is not None:
                    # Delete existing analysis results
                    self._cascade_delete_analysis_results(
                        session=session,
                        experiment=experiment,
                        detection_settings_id=detection_settings.id,
                    )

                # Save FOV results with detection_settings_id
                positions_processed = []
                for fov_result in fov_results:
                    commit_fov_result(
                        session, experiment, fov_result, detection_settings.id
                    )
                    positions_processed.append(fov_result.position_index)

                # Create/update AnalysisResult to track detection-only run
                if positions_processed and detection_settings.id is not None:
                    from datetime import datetime

                    from sqlalchemy import desc

                    from cali.sqlmodel._model import AnalysisResult

                    # Check if ANY AnalysisResult exists for this detection
                    # (detection-only OR full analysis), ordered by most recent
                    existing_any_result = session.exec(
                        select(AnalysisResult)
                        .where(
                            AnalysisResult.experiment == experiment.id,
                            AnalysisResult.detection_settings == detection_settings.id,
                        )
                        .order_by(desc(AnalysisResult.created_at))  # type: ignore
                    ).first()

                    if (
                        existing_any_result is not None
                        and existing_any_result.analysis_settings is not None
                    ):
                        # Full AnalysisResult exists - update its timestamp
                        existing_any_result.created_at = datetime.now()
                        session.add(existing_any_result)
                        session.commit()
                        cali_logger.info(
                            f"⏭️  Skipping detection-only AnalysisResult creation - "
                            f"updated AnalysisResult ID {existing_any_result.id} "
                            f"timestamp (DetectionSettings={detection_settings.id})"
                        )
                    elif existing_any_result is not None:
                        # Merge positions into existing detection-only result
                        existing_pos = set(existing_any_result.positions_analyzed or [])
                        new_positions = set(positions_processed)
                        merged_positions = sorted(existing_pos | new_positions)
                        existing_any_result.positions_analyzed = merged_positions
                        session.add(existing_any_result)
                        session.commit()
                        cali_logger.info(
                            f"🔄 Updated detection-only AnalysisResult ID "
                            f"{existing_any_result.id} to include "
                            f"positions {merged_positions}"
                        )
                    else:
                        # Create new detection-only AnalysisResult
                        if experiment.id is not None:
                            detection_result = AnalysisResult(
                                experiment=experiment.id,
                                detection_settings=detection_settings.id,
                                analysis_settings=None,  # Detection-only
                                positions_analyzed=positions_processed,
                            )
                            session.add(detection_result)
                            session.commit()
                            session.refresh(detection_result)
                            cali_logger.info(
                                f"📊 Created detection-only AnalysisResult ID "
                                f"{detection_result.id} (DetectionSettings="
                                f"{detection_settings.id}, "
                                f"positions={positions_processed})"
                            )
            engine.dispose(close=True)

    def _run_cellpose_detection(
        self,
        experiment: Experiment,
        global_position_indices: Sequence[int],
        model: CellposeModel,
        diameter: float | None,
        cellprob_threshold: float,
        flow_threshold: float,
        batch_size: int,
        min_size: int,
        normalize: bool,
        overwrite: bool,
    ) -> list[FOV]:
        """Internal method to run Cellpose detection and return FOV objects.

        Returns
        -------
        list[FOV]
            List of FOV objects with ROIs and Masks, ready to be committed
        """
        from sqlmodel import select

        # DATABASE DO NOT EXISTS
        if not Path(experiment.db_path).exists():
            save_experiment_to_database(experiment)

        # DATABASE EXISTS
        engine = create_engine(f"sqlite:///{experiment.db_path}", echo=False)
        with Session(engine) as session:
            # if database does exist but the overwrite flag is True, just overwrite
            if overwrite:
                save_experiment_to_database(experiment, overwrite=True)
            else:
                # if database does exist but the the experiment.id is either None or
                # different than the one in the database, raise ValueError.
                db_exp = cast("Experiment", session.exec(select(Experiment)).first())
                if experiment.id is None or experiment.id != db_exp.id:
                    msg = (
                        "The provided Experiment must have an ID matching the one "
                        f"in the database (ID: {db_exp.id} vs {experiment.id}). Either "
                        f"set the `Experiment.id` to {db_exp.id}, use a different "
                        "`Experiment.database_name` or `Experiment.analysis_path` or"
                        "`set the overwrite flag to `True` to overwrite the database."
                    )
                    cali_logger.error(msg)
                    engine.dispose(close=True)
                    raise ValueError(msg)
        engine.dispose(close=True)

        # load data
        self._data = load_data(experiment.data_path)

        if experiment.id is None:
            msg = "Experiment must have an ID before running detection"
            raise ValueError(msg)

        # Load all images for batch processing
        all_images = []
        all_metadata = []
        all_pos_indices = []

        cali_logger.info("Loading images for batch processing...")
        for pos_idx in global_position_indices:
            if self._check_for_abort_requested():
                cali_logger.info("Detection cancelled during image loading")
                return []

            data, meta = self._data.isel(p=pos_idx, metadata=True)

            # Preprocess data: max projection from half to end of stack
            if data.ndim == 3:  # (t, y, x)
                data_half_to_end = data[data.shape[0] // 2 :, :, :]
                image = data_half_to_end.max(axis=0)
            else:  # already 2D
                image = data

            all_images.append(image)
            all_metadata.append(meta)
            all_pos_indices.append(pos_idx)

        if self._check_for_abort_requested():
            return []

        # Process in batches
        cali_logger.info(
            f"Processing {len(all_images)} images in batches of {batch_size}"
        )
        all_masks = self._batch_process(
            model=model,
            images=all_images,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            min_size=min_size,
            normalize=normalize,
        )

        if self._check_for_abort_requested():
            return []

        # Create FOV objects
        fov_results = []

        for pos_idx, meta, masks_2d in zip(all_pos_indices, all_metadata, all_masks):
            if self._check_for_abort_requested():
                cali_logger.info("Detection cancelled during FOV creation")
                return fov_results

            fov_result = self._create_fov_with_rois(pos_idx, meta, masks_2d)

            if fov_result:
                fov_results.append(fov_result)

        return fov_results

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_event.is_set()

    def _batch_process(
        self,
        model: CellposeModel,
        images: list[np.ndarray],
        diameter: float | None,
        cellprob_threshold: float,
        flow_threshold: float,
        batch_size: int,
        min_size: int,
        normalize: bool,
    ) -> list[np.ndarray]:
        """Process images in batches using Cellpose.

        Returns
        -------
        list[np.ndarray]
            List of 2D label masks, one per image
        """
        from cellpose.utils import fill_holes_and_remove_small_masks

        all_masks = []
        n_batches = (len(images) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Running Cellpose"):
            if self._check_for_abort_requested():
                break

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images))
            batch_images = images[start_idx:end_idx]

            # Run Cellpose on batch
            masks, _, _ = model.eval(
                batch_images,
                diameter=diameter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                normalize=normalize,
                batch_size=len(batch_images),
            )

            # Post-process masks
            for mask in masks:
                mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)
                all_masks.append(mask)

        return all_masks

    def _create_fov_with_rois(
        self,
        global_pos_idx: int,
        meta: list[dict],
        masks_2d: np.ndarray,
    ) -> FOV | None:
        """Create FOV with ROIs and Masks from segmentation results.

        Parameters
        ----------
        experiment : Experiment
            Parent experiment
        global_pos_idx : int
            Position index
        meta : list[dict]
            Metadata for this position
        masks_2d : np.ndarray
            2D label mask from Cellpose

        Returns
        -------
        FOV | None
            FOV object with ROIs and Masks, ready to commit
        """
        # Get FOV name from metadata
        fov_name = _get_fov_name(EVENT_KEY, meta, global_pos_idx)

        # Get unique label values (excluding background 0)
        label_values = np.unique(masks_2d)
        label_values = label_values[label_values > 0]

        if len(label_values) == 0:
            cali_logger.warning(f"No cells detected in {fov_name}")
            return None

        # Create FOV (well association will be handled by commit_detection_result)
        fov = FOV(
            name=fov_name,
            position_index=global_pos_idx,
            fov_number=global_pos_idx,
            rois=[],
        )

        # Create ROIs with masks
        for label_value in label_values:
            if self._check_for_abort_requested():
                return None

            # Create binary mask for this ROI
            roi_mask_binary = masks_2d == label_value

            # Convert to coordinates
            mask_coords, mask_shape = mask_to_coordinates(roi_mask_binary)

            # Create Mask object
            mask_obj = Mask(
                coords_y=mask_coords[0],
                coords_x=mask_coords[1],
                height=mask_shape[0],
                width=mask_shape[1],
                mask_type="roi",
            )

            # Create ROI with mask
            # Note: fov_id will be set when the FOV is persisted
            # The relationship will handle the connection
            roi = ROI(
                label_value=int(label_value),
                active=None,  # Will be determined during analysis
                stimulated=None,  # Will be determined during analysis
                roi_mask=mask_obj,
                fov_id=0,  # Placeholder - will be set by relationship
            )

            fov.rois.append(roi)
        return fov

    def _get_or_create_detection_settings(
        self, session: Session, detection_settings: DetectionSettings
    ) -> DetectionSettings:
        """Get existing or create new DetectionSettings in database.

        Parameters
        ----------
        session : Session
            Database session
        detection_settings : DetectionSettings
            Detection settings to find or create

        Returns
        -------
        DetectionSettings
            Existing or newly created settings with ID
        """
        if detection_settings.id is None:
            # Check if identical settings already exist in database
            try:
                all_settings = session.exec(select(DetectionSettings)).all()
                existing_settings = None
                for candidate in all_settings:
                    if detection_settings == candidate:
                        existing_settings = candidate
                        break
            except Exception:
                existing_settings = None

            # Found duplicate - use the existing one
            if existing_settings is not None:
                return existing_settings

            # Create new settings
            session.add(detection_settings)
            session.commit()
            session.refresh(detection_settings)
            return detection_settings
        else:
            # Settings already has an ID - check if it exists
            all_settings_ids = [
                s.id for s in session.exec(select(DetectionSettings)).all()
            ]
            if detection_settings.id not in all_settings_ids:
                session.add(detection_settings)
                session.commit()
                session.refresh(detection_settings)
                return detection_settings
            else:
                # Get the existing settings from database
                existing = session.get(DetectionSettings, detection_settings.id)
                if existing is not None:
                    return existing
                return detection_settings

    def _cascade_delete_analysis_results(
        self,
        session: Session,
        experiment: Experiment,
        detection_settings_id: int,
    ) -> None:
        """Delete all AnalysisResults using these DetectionSettings.

        This cascades to delete Traces, DataAnalysis, and other related data.
        ROIs will be replaced during the new detection run.

        Parameters
        ----------
        session : Session
            Database session
        experiment : Experiment
            Current experiment
        detection_settings_id : int
            ID of DetectionSettings to clear
        """
        from cali.sqlmodel._model import AnalysisResult

        # Find all AnalysisResults using these detection settings
        results_to_delete = session.exec(
            select(AnalysisResult).where(
                AnalysisResult.experiment == experiment.id,
                AnalysisResult.detection_settings == detection_settings_id,
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
