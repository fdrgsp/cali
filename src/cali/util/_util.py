import logging
from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from cali._constants import TS, ZR
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.sqlmodel._model import (
    FOV,
    Experiment,
    Plate,
    Well,
)

cali_logger = logging.getLogger("cali_logger")


def load_data(data_path: str | Path) -> TensorstoreZarrReader | OMEZarrReader:
    """Load data from the given path using the appropriate reader."""
    data_path = str(data_path)
    # select which reader to use for the datastore
    if data_path.endswith(TS):
        # read tensorstore
        return TensorstoreZarrReader(data_path)
    elif data_path.endswith(ZR):
        # read ome zarr
        return OMEZarrReader(data_path)
    else:
        msg = f"Unsupported data format for path: {data_path}"
        cali_logger.error(msg)
        raise ValueError(msg)


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
) -> None:
    """Commit FOV result to database.

    Handles both detection results (ROIs with masks only) and analysis results
    (ROIs with masks, traces, and analysis data). The function works for both
    cases because it simply replaces all ROIs, regardless of their content.

    Parameters
    ----------
    session : Session
        Database session
    experiment : Experiment
        Parent experiment
    fov_result : FOV
        FOV with ROIs to commit
    detection_settings_id : int | None
        Detection settings ID to assign to all ROIs (required for detection,
        optional for analysis which reads it from existing ROIs)
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
            # DETECTION MODE: Replace ROIs with same detection_settings_id
            # Delete old ROIs that match this detection_settings_id
            for old_roi in list(existing_fov.rois):
                if old_roi.detection_settings_id == detection_settings_id:
                    session.delete(old_roi)
            session.flush()

            # Add new ROIs from detection
            # IMPORTANT: Iterate over a copy of the list because SQLAlchemy modifies
            # fov_result.rois when we set roi.fov = existing_fov
            for roi in list(fov_result.rois):
                roi.fov_id = existing_fov.id
                roi.fov = existing_fov
                roi.detection_settings_id = detection_settings_id
                session.add(roi)
        else:
            # ANALYSIS MODE: Don't create new ROIs, only attach traces/analysis
            # to existing ROIs
            # Match ROIs by (label_value, detection_settings_id) and update
            for new_roi in fov_result.rois:
                # Find matching existing ROI
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

                    # Add traces and data_analysis to existing ROI
                    for trace in new_roi.traces_history:
                        trace.roi_id = matching_roi.id
                        trace.roi = matching_roi
                        session.add(trace)

                    for data_analysis in new_roi.data_analysis_history:
                        data_analysis.roi_id = matching_roi.id
                        data_analysis.roi = matching_roi
                        session.add(data_analysis)
                else:
                    cali_logger.warning(
                        f"No matching ROI found for label={new_roi.label_value}, "
                        f"detection_settings={new_roi.detection_settings_id} "
                        f"in FOV {existing_fov.name}"
                    )
    else:
        # New FOV - link to well and add
        # Set detection_settings_id on all ROIs if provided
        if detection_settings_id is not None:
            for roi in fov_result.rois:
                roi.detection_settings_id = detection_settings_id
        fov_result.well_id = well.id
        session.add(fov_result)

    session.commit()
