from pathlib import Path

import numpy as np
import tifffile
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine, select
from tqdm import tqdm

from cali._constants import TS, ZR
from cali.logger import cali_logger
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.sqlmodel._model import (
    FOV,
    ROI,
    DataAnalysis,
    Experiment,
    Plate,
    Traces,
    Well,
)


def load_data_from_path(
    data_path: str | Path,
) -> TensorstoreZarrReader | OMEZarrReader | None:
    """Load data from the given path using the appropriate reader.

    Parameters
    ----------
    data_path : str | Path
        Path to the data directory or file

    Returns
    -------
    TensorstoreZarrReader | OMEZarrReader | None
        The appropriate reader for the data format, or None if unsupported
    """
    cali_logger.info(f"💿 Loading data from path: {data_path}")
    data_path_str = str(data_path)

    # read tensorstore from micromanager-gui package
    if data_path_str.endswith(TS):
        return TensorstoreZarrReader(data_path)

    # read ome zarr from micromanager-gui package
    elif data_path_str.endswith(ZR):
        return OMEZarrReader(data_path)

    return None
    # msg = f"Unsupported data format for path: {data_path}"
    # cali_logger.error(msg)
    # raise ValueError(msg)


def mask_to_coordinates(
    mask: np.ndarray,
) -> tuple[tuple[list[int], list[int]], tuple[int, int]]:
    """Convert a 2D boolean mask to sparse coordinates.

    Args:
        mask: 2D boolean numpy array

    Returns
    -------
        Tuple of ((y_coords, x_coords), (height, width))
    """
    y_coords, x_coords = np.where(mask)
    y_coords_list: list[int] = [int(y) for y in y_coords]
    x_coords_list: list[int] = [int(x) for x in x_coords]
    return ((y_coords_list, x_coords_list), (mask.shape[0], mask.shape[1]))


def coordinates_to_mask(
    coordinates: tuple[list[int], list[int]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Convert sparse coordinates back to a 2D boolean mask.

    Args:
        coordinates: Tuple of (y_coords, x_coords) lists
        shape: Tuple of (height, width)

    Returns
    -------
        2D boolean numpy array
    """
    mask = np.zeros(shape, dtype=bool)
    y_coords, x_coords = coordinates
    mask[y_coords, x_coords] = True
    return mask


def commit_fov_result(
    session: Session,
    experiment: Experiment,
    fov_result: FOV,
    detection_settings_id: int | None = None,
    commit: bool = True,
) -> None:
    """Commit FOV result to database.

    Handles both detection results (ROIs with masks only) and analysis results
    (ROIs with masks, traces, and analysis data).

    For detection: Creates new ROIs with detection_settings_id set.
    For analysis: Adds traces/analysis to existing ROIs matching by label_value
    and detection_settings_id.

    Parameters
    ----------
    session : Session
        Database session
    experiment : Experiment
        Parent experiment
    fov_result : FOV
        FOV with ROIs to commit
    detection_settings_id : int | None
        Detection settings ID to assign to new ROIs (required for detection,
        optional for analysis which reads it from existing ROIs)
    commit : bool
        Whether to commit immediately (True) or let caller handle batched commits
        (False). Default True.
    """
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
        # Update FOV metadata
        existing_fov.position_index = fov_result.position_index
        existing_fov.fov_number = fov_result.fov_number
        existing_fov.fov_metadata = fov_result.fov_metadata
        existing_fov.well_id = well.id

        if detection_settings_id is not None:
            # DETECTION MODE: Add new ROIs with detection_settings_id
            # Multiple detections can coexist on same FOV

            # Check if ROIs with this detection_settings_id already exist
            # to avoid duplicates if run multiple times
            existing_rois_map = {
                r.label_value: r
                for r in existing_fov.rois
                if r.detection_settings_id == detection_settings_id
            }

            # Create new ROI instances to avoid cascade-adding fov_result
            # (ROIs from detection have roi.fov = fov_result, which would
            # cause fov_result to be added to session via relationship cascade)
            for old_roi in fov_result.rois:
                if old_roi.label_value in existing_rois_map:
                    continue

                new_roi = ROI(
                    label_value=old_roi.label_value,
                    active=old_roi.active,
                    stimulated=old_roi.stimulated,
                    fov_id=existing_fov.id,
                    detection_settings_id=detection_settings_id,
                    roi_mask=old_roi.roi_mask,
                )
                session.add(new_roi)

            # Flush to assign IDs to new ROIs
            session.flush()
        else:
            # ANALYSIS MODE: Don't create new ROIs, only attach traces/analysis
            # to existing ROIs
            # Match ROIs by label_value and detection_settings_id
            for new_roi in fov_result.rois:
                # Find matching existing ROI by label_value and detection_settings_id
                matching_roi = None
                for existing_roi in existing_fov.rois:
                    if (
                        existing_roi.label_value == new_roi.label_value
                        and existing_roi.detection_settings_id
                        == new_roi.detection_settings_id
                    ):
                        matching_roi = existing_roi
                        break

                if matching_roi:
                    # Update ROI properties from analysis
                    matching_roi.active = new_roi.active
                    matching_roi.stimulated = new_roi.stimulated
                    # Update cell size if provided (set during extraction)
                    if new_roi.cell_size is not None:
                        matching_roi.cell_size = new_roi.cell_size
                    if new_roi.cell_size_units is not None:
                        matching_roi.cell_size_units = new_roi.cell_size_units

                    # Check if traces/analysis already exist for this analysis_result_id
                    # to avoid duplicates if analysis is run multiple times
                    existing_analysis_ids = set()
                    for trace in new_roi.traces_history:
                        if trace.analysis_result_id is not None:
                            # Check if trace with this analysis_result_id already exists
                            existing_trace = next(
                                (
                                    t
                                    for t in matching_roi.traces_history
                                    if t.analysis_result_id == trace.analysis_result_id
                                ),
                                None,
                            )
                            if existing_trace:
                                # Skip - trace already exists
                                existing_analysis_ids.add(trace.analysis_result_id)
                                continue

                        # Clear the trace's ROI reference first to avoid cascading the
                        # wrong ROI
                        # (trace.roi might point to new_roi which has same ID as
                        # matching_roi)
                        trace.roi = None  # type: ignore[assignment]
                        trace.roi_id = None

                        # Check if trace with this ID already exists in session
                        # (avoids "already present in this session" error)
                        if trace.id is not None:
                            existing_in_session = session.get(Traces, trace.id)
                            if existing_in_session is not None:
                                # Use existing trace from session instead
                                trace = existing_in_session
                        else:
                            # New trace without ID - add to session
                            session.add(trace)

                        # Now set to the correct matching_roi
                        trace.roi_id = matching_roi.id
                        matching_roi.traces_history.append(trace)

                    # Only add new data_analysis if not already exists
                    for data_analysis in new_roi.data_analysis_history:
                        if data_analysis.analysis_result_id is not None:
                            existing_data_analysis = next(
                                (
                                    da
                                    for da in matching_roi.data_analysis_history
                                    if da.analysis_result_id
                                    == data_analysis.analysis_result_id
                                ),
                                None,
                            )
                            if existing_data_analysis:
                                # Skip - data_analysis already exists
                                continue

                        # Clear the data_analysis's ROI reference first to avoid
                        # cascading the wrong ROI
                        data_analysis.roi = None  # type: ignore[assignment]
                        data_analysis.roi_id = None

                        # Check if data_analysis with this ID already exists in session
                        # (avoids "already present in this session" error)
                        if data_analysis.id is not None:
                            existing_in_session = session.get(
                                DataAnalysis, data_analysis.id
                            )
                            if existing_in_session is not None:
                                # Use existing data_analysis from session instead
                                data_analysis = existing_in_session
                        else:
                            # New data_analysis without ID - add to session
                            session.add(data_analysis)

                        # Now set to the correct matching_roi
                        data_analysis.roi_id = matching_roi.id
                        matching_roi.data_analysis_history.append(data_analysis)
                else:
                    cali_logger.warning(
                        f"No matching ROI found for label={new_roi.label_value} "
                        f"detection_settings_id={new_roi.detection_settings_id} "
                        f"in FOV {existing_fov.name}"
                    )
    else:
        # New FOV - link to well and add
        # Set detection_settings_id on each ROI if provided (detection mode)
        if detection_settings_id is not None:
            for roi in fov_result.rois:
                roi.detection_settings_id = detection_settings_id
        fov_result.well_id = well.id
        session.add(fov_result)

    if commit:
        session.commit()


def update_fovs_in_database(
    db_path: Path | str | Engine,
    fovs: list[FOV] | FOV,
    *,
    echo: bool = False,
) -> None:
    """Update FOVs in database, saving all related ROIs, Traces, and DataAnalysis.

    This function enables manual pipeline workflows where you can run detection,
    extraction, and analysis steps separately and save results after each step.

    **Important**: After calling this function, reload the FOVs from the database
    before passing them to the next pipeline step. This ensures relationships
    are properly populated.

    Parameters
    ----------
    db_path : Path | str | Engine
        Path to SQLite database file or existing SQLAlchemy engine
    fovs : list[FOV] | FOV
        Single FOV or list of FOVs to update in database
    echo : bool, optional
        Whether to enable SQLAlchemy engine echo for debugging, by default False
    """
    # Handle single FOV or list
    fov_list = [fovs] if isinstance(fovs, FOV) else fovs

    # Get or create engine
    if isinstance(db_path, Engine):
        engine = db_path
        should_dispose = False
    else:
        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=echo,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )
        should_dispose = True

    try:
        with Session(engine) as session:
            for fov in fov_list:
                # Transfer temporary attributes to relationships
                # (ExtractionRunner stores traces in _new_traces temporarily)
                for roi in fov.rois:
                    if hasattr(roi, "_new_traces"):
                        for trace in roi._new_traces:
                            roi.traces_history.append(trace)
                        delattr(roi, "_new_traces")

                    if hasattr(roi, "_new_data_analysis"):
                        for data_analysis in roi._new_data_analysis:
                            roi.data_analysis_history.append(data_analysis)
                        delattr(roi, "_new_data_analysis")

                # Load existing FOV from database by position_index to get the ID
                from sqlmodel import select

                db_fov = session.exec(
                    select(FOV).where(FOV.position_index == fov.position_index)
                ).first()

                if db_fov:
                    # Update existing - set the ID so merge updates instead of inserting
                    fov.id = db_fov.id
                    fov.well_id = db_fov.well_id

                    # Match and update ROIs by label_value
                    for roi in fov.rois:
                        db_roi = next(
                            (
                                r
                                for r in db_fov.rois
                                if r.label_value == roi.label_value
                            ),
                            None,
                        )
                        if db_roi:
                            roi.id = db_roi.id
                            roi.fov_id = db_roi.fov_id

                session.merge(fov)
            session.commit()
    finally:
        if should_dispose:
            engine.dispose(close=True)


def load_fovs_from_database(
    db_path: Path | str | Engine,
    position_indices: list[int] | int | None = None,
    *,
    echo: bool = False,
) -> list["FOV"]:
    """Load FOVs from database by position indices.

    This function is typically used after update_fovs_in_database() to reload
    FOVs with all relationships properly populated before passing to the next
    pipeline step.

    Parameters
    ----------
    db_path : Path | str | Engine
        Path to SQLite database file or existing SQLAlchemy engine
    position_indices : list[int] | int | None
        Single position index or list of position indices to load. If None,
        loads all FOVs in database, by default None
    echo : bool, optional
        Whether to enable SQLAlchemy engine echo for debugging, by default False

    Returns
    -------
    list[FOV]
        List of FOVs with all relationships loaded and detached from session

    Example
    -------
    >>> from cali.util import load_fovs_from_database, update_fovs_in_database
    >>>
    >>> # Manual pipeline workflow
    >>> # 1. Detection
    >>> fovs = detection_runner.run(dataset, detection_settings, [17, 18])
    >>> update_fovs_in_database("results.cali", fovs)
    >>>
    >>> # 2. Reload for extraction
    >>> fovs = load_fovs_from_database("results.cali", [17, 18])
    >>> fovs = extraction_runner.run(dataset, extraction_settings, fovs)
    >>> update_fovs_in_database("results.cali", fovs)
    >>>
    >>> # 3. Reload for analysis
    >>> fovs = load_fovs_from_database("results.cali", [17, 18])
    >>> fovs = analysis_runner.run(fovs, analysis_settings)
    >>> update_fovs_in_database("results.cali", fovs)
    """
    from sqlalchemy.engine import Engine
    from sqlmodel import Session, create_engine, select

    from cali.sqlmodel._model import FOV

    # Handle single position index or list
    pos_list = None
    if position_indices is not None:
        pos_list = (
            [position_indices]
            if isinstance(position_indices, int)
            else position_indices
        )

    # Get or create engine
    if isinstance(db_path, Engine):
        engine = db_path
        should_dispose = False
    else:
        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=echo,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )
        should_dispose = True

    try:
        with Session(engine) as session:
            from sqlalchemy.orm import selectinload

            # Eager load all relationships to avoid lazy-loading after detachment
            query = select(FOV).options(
                selectinload(FOV.rois).selectinload(ROI.traces_history),
                selectinload(FOV.rois).selectinload(ROI.data_analysis_history),
            )

            if pos_list is not None:
                query = query.where(FOV.position_index.in_(pos_list))  # type: ignore

            fovs = list(session.exec(query).all())
            session.expunge_all()  # Detach from session
            return fovs
    finally:
        if should_dispose:
            engine.dispose(close=True)


def save_labeled_images(
    db_path: Path | str | Engine,
    output_dir: str | Path,
    *,
    position_indices: list[int] | int | None = None,
    detection_settings_id: int | None = None,
    overwrite: bool = False,
    echo: bool = False,
) -> None:
    """Save labeled images for FOVs loaded from database.

    Loads FOVs from database by position indices and saves labeled images.

    Parameters
    ----------
    db_path : Path | str | Engine
        Path to SQLite database file or existing SQLAlchemy engine
    output_dir : str | Path
        Directory to save labeled images
    position_indices : list[int] | int | None
        Single position index or list of position indices to load. If None,
        loads all FOVs in database, by default None
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    echo : bool, optional
        Whether to enable SQLAlchemy engine echo for debugging, by default False
    detection_settings_id : int | None, optional
        If provided, only include ROIs with this detection_settings_id.
        If None, include all ROIs, by default None
    """
    fovs = load_fovs_from_database(db_path, position_indices, echo=echo)
    save_labeled_images_from_fovs(
        fovs,
        output_dir,
        overwrite=overwrite,
        detection_settings_id=detection_settings_id,
    )


def save_labeled_images_from_fovs(
    fovs: list[FOV] | FOV,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    detection_settings_id: int | None = None,
) -> None:
    """Save labeled images for FOVs to disk.

    Creates labeled TIFF images where each ROI is assigned its label_value
    as pixel intensity. Background pixels are 0.

    Parameters
    ----------
    fovs : list[FOV] | FOV
        Single FOV or list of FOVs to save labeled images for
    output_dir : str | Path
        Directory to save labeled images
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    detection_settings_id : int | None, optional
        If provided, only include ROIs with this detection_settings_id.
        If None, include all ROIs, by default None

    Raises
    ------
    FileExistsError
        If output file exists and overwrite=False
    ValueError
        If ROI mask data is missing or invalid
    """
    # Handle single FOV or list
    fov_list = [fovs] if isinstance(fovs, FOV) else fovs

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for fov in tqdm(fov_list, desc="Saving labeled images"):
        if not fov.rois:
            cali_logger.warning(f"⚠️ No ROIs found in FOV {fov.name}, skipping")
            continue

        # Get image dimensions from first ROI mask
        first_roi = fov.rois[0]
        if not first_roi.roi_mask:
            cali_logger.warning(
                f"⚠️ ROI {first_roi.label_value} in {fov.name} has no mask, skipping FOV"
            )
            continue

        height = first_roi.roi_mask.height
        width = first_roi.roi_mask.width

        if height is None or width is None:
            cali_logger.warning(
                f"⚠️ Invalid mask dimensions for FOV {fov.name}, skipping"
            )
            continue

        # Create labeled image
        labeled_image = np.zeros((height, width), dtype=np.uint16)

        # Filter ROIs by detection_settings_id if provided
        filtered_rois = fov.rois
        if detection_settings_id is not None:
            filtered_rois = [
                roi
                for roi in fov.rois
                if roi.detection_settings_id == detection_settings_id
            ]

        for roi in filtered_rois:
            if not roi.roi_mask:
                cali_logger.warning(
                    f"⚠️ ROI {roi.label_value} in {fov.name} has no mask, skipping"
                )
                continue

            # Convert mask coordinates to boolean mask
            coords_y = roi.roi_mask.coords_y
            coords_x = roi.roi_mask.coords_x

            if coords_y is None or coords_x is None:
                cali_logger.warning(
                    f"⚠️ ROI {roi.label_value} in {fov.name} "
                    "has invalid mask coordinates, skipping"
                )
                continue

            # Set pixels for this ROI to its label value
            labeled_image[coords_y, coords_x] = roi.label_value

        # Save labeled image
        filename = f"{fov.name}_labeled.tif"
        output_file = output_path / filename

        if output_file.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {output_file} exists and overwrite=False"
            )

        tifffile.imwrite(output_file, labeled_image)
