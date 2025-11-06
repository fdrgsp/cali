"""Migration helper: Import existing JSON analysis data into SQLModel database.

This module provides functions to migrate calcium imaging analysis data
from JSON format to the new SQLModel database format.

Example
-------
>>> from pathlib import Path
>>> from migrate_json_to_db import load_analysis_from_json
>>> analysis_dir = Path("tests/test_data/spontaneous/spont_analysis")
>>> experiment = load_analysis_from_json(analysis_dir, experiment_name="Spontaneous")
>>> print(f"Loaded {len(experiment.plate.wells)} wells")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from sqlmodel import Session, create_engine

from cali._plate_viewer._util import ROIData

from ._models import (
    FOV,
    ROI,
    AnalysisSettings,
    Condition,
    DataAnalysis,
    Experiment,
    Mask,
    Plate,
    Traces,
    Well,
    create_db_and_tables,
)

if TYPE_CHECKING:
    from useq import WellPlate


def load_analysis_from_json(analysis_dir: Path, useq_plate: WellPlate) -> Experiment:
    """Load analysis data from JSON directory into SQLModel objects.

    This function reads all JSON files in the analysis directory and creates
    a complete Experiment object with all related entities (Plate, Wells,
    FOVs, ROIs, Conditions, AnalysisSettings, Masks, Traces, DataAnalysis).

    The function does NOT save to a database - it returns an in-memory object
    tree that can be added to a SQLModel session.

    Parameters
    ----------
    analysis_dir : Path
        Directory containing JSON analysis files (e.g., spont_analysis)
    useq_plate : WellPlate
        useq-schema WellPlate object defining the plate used

    Returns
    -------
    Experiment
        Complete experiment object with all relationships populated

    Example
    -------
    >>> from pathlib import Path
    >>> from sqlmodel import Session, create_engine
    >>> from models import create_tables
    >>> import useq
    >>>
    >>> # Load from JSON
    >>> analysis_dir = Path("tests/test_data/spontaneous/spont_analysis")
    >>> plate = useq.WellPlate.from_str("96-well")
    >>> experiment = load_analysis_from_json(analysis_dir, useq_plate=plate)
    >>>
    >>> # Save to database
    >>> engine = create_engine("sqlite:///test.db")
    >>> create_tables(engine)
    >>> with Session(engine) as session:
    ...     session.add(experiment)
    ...     session.commit()
    """
    experiment_name = analysis_dir.parent.name

    # 1. Create experiment
    experiment = Experiment(
        name=experiment_name,
        description=f"Imported from {analysis_dir}",
        data_path=str(
            analysis_dir.parent / f"{analysis_dir.parent.name}.tensorstore.zarr"
        ),
        labels_path=str(analysis_dir.parent / f"{analysis_dir.parent.name}_labels"),
        analysis_path=str(analysis_dir),
    )

    # 2. Create plate
    plate = Plate(
        experiment_id=experiment.id,
        experiment=experiment,
        name=useq_plate.name,
        plate_type=useq_plate.name,
        rows=useq_plate.rows,
        columns=useq_plate.columns,
    )

    # 3. Load and create conditions
    genotype_map_path = analysis_dir / "genotype_plate_map.json"
    treatment_map_path = analysis_dir / "treatment_plate_map.json"

    genotype_map = load_plate_map(genotype_map_path)
    treatment_map = load_plate_map(treatment_map_path)

    conditions: dict[str, Condition] = {}

    # Create genotype conditions
    for well_data in genotype_map.values():
        name = well_data["name"]
        if name not in conditions:
            cond = Condition(
                name=name,
                condition_type="genotype",
                color=well_data["color"],
            )
            conditions[name] = cond

    # Create treatment conditions
    for well_data in treatment_map.values():
        name = well_data["name"]
        if name not in conditions:
            cond = Condition(
                name=name,
                condition_type="treatment",
                color=well_data["color"],
            )
            conditions[name] = cond

    # 4. Load analysis settings
    settings_path = analysis_dir / "settings.json"
    analysis_settings = None

    if settings_path.exists():
        with open(settings_path) as f:
            settings_data = json.load(f)

        analysis_settings = AnalysisSettings(
            experiment_id=experiment.id,
            experiment=experiment,
            dff_window=settings_data.get("dff_window", 30),
            decay_constant=settings_data.get("decay constant", 0.0),
            peaks_height_value=settings_data.get("peaks_height_value", 3.0),
            peaks_height_mode=settings_data.get("peaks_height_mode", "multiplier"),
            peaks_distance=settings_data.get("peaks_distance", 2),
            peaks_prominence_multiplier=settings_data.get(
                "peaks_prominence_multiplier", 1.0
            ),
            calcium_network_threshold=settings_data.get(
                "calcium_network_threshold", 90.0
            ),
            spike_threshold_value=settings_data.get("spike_threshold_value", 1.0),
            spike_threshold_mode=settings_data.get(
                "spike_threshold_mode", "multiplier"
            ),
            burst_threshold=settings_data.get("burst_threshold", 30.0),
            burst_min_duration=settings_data.get("burst_min_duration", 3),
            burst_gaussian_sigma=settings_data.get("burst_gaussian_sigma", 2.0),
            spikes_sync_cross_corr_lag=settings_data.get(
                "spikes_sync_cross_corr_lag", 5
            ),
            calcium_sync_jitter_window=settings_data.get(
                "calcium_sync_jitter_window", 5
            ),
            neuropil_inner_radius=settings_data.get("neuropil_inner_radius", 0),
            neuropil_min_pixels=settings_data.get("neuropil_min_pixels", 0),
            neuropil_correction_factor=settings_data.get(
                "neuropil_correction_factor", 0.0
            ),
            led_power_equation=settings_data.get("led_power_equation") or None,
        )

    # 5. Process JSON files
    json_files = [
        f
        for f in analysis_dir.glob("*.json")
        if f.name
        not in [
            "settings.json",
            "genotype_plate_map.json",
            "treatment_plate_map.json",
        ]
        and not f.name.startswith("._")  # Skip macOS resource fork files
    ]

    wells_created: dict[str, Well] = {}
    total_rois = 0

    for json_file in json_files:
        # Parse FOV name
        fov_name = json_file.stem
        well_name = fov_name.split("_")[0]

        # Create or get well
        if well_name not in wells_created:
            row, col = parse_well_name(well_name)

            # Create well with conditions
            well = Well(
                plate_id=plate.id,
                plate=plate,  # Use relationship
                name=well_name,
                row=row,
                column=col,
                conditions=[],  # Will populate below
            )

            # Add conditions to well
            if well_name in genotype_map:
                genotype_cond = conditions[genotype_map[well_name]["name"]]
                well.conditions.append(genotype_cond)

            if well_name in treatment_map:
                treatment_cond = conditions[treatment_map[well_name]["name"]]
                well.conditions.append(treatment_cond)

            wells_created[well_name] = well

        well = wells_created[well_name]

        # Create FOV
        if "_p" in fov_name:
            position_index = int(fov_name.split("_p")[-1])
        else:
            position_index = 0

        # Extract fov_number
        parts = fov_name.split("_")
        if len(parts) >= 2:
            fov_number = int(parts[1])
        else:
            fov_number = 0

        fov = FOV(
            well=well,  # Use relationship
            name=fov_name,
            position_index=position_index,
            fov_number=fov_number,
        )

        # Load and create ROIs
        with open(json_file) as f:
            roi_dict = json.load(f)

        roi_count = 0
        for roi_label, roi_data_dict in roi_dict.items():
            if not roi_label.isdigit():
                continue  # Skip non-ROI data

            try:
                roi_data = ROIData(**roi_data_dict)

                # Use roi_from_roi_data helper
                roi, trace, data_analysis, roi_mask, neuropil_mask = roi_from_roi_data(
                    roi_data,
                    fov_id=fov.id or 0,  # Placeholder, will be set via relationship
                    label_value=int(roi_label),
                    settings_id=(analysis_settings.id if analysis_settings else None),
                )

                # Since we're not in DB session, manually set relationships
                roi.fov = fov
                roi.analysis_settings = analysis_settings
                roi.roi_mask = roi_mask
                roi.neuropil_mask = neuropil_mask
                roi.traces = trace
                roi.data_analysis = data_analysis

                # Add ROI to FOV
                fov.rois.append(roi)

                roi_count += 1
                total_rois += 1
            except Exception as e:
                print(f"  ⚠ Error importing ROI {roi_label} from ")
                print(f"    {json_file.name}: {e}")
                continue

    return experiment


def parse_well_name(well_name: str) -> tuple[int, int]:
    """Parse well name like 'B5' into (row, column) indices.

    Parameters
    ----------
    well_name : str
        Well name (e.g., 'B5', 'A1')

    Returns
    -------
    tuple[int, int]
        (row, column) - Zero-indexed row and column

    Raises
    ------
    ValueError
        If well_name is not in the expected format
    """
    if not well_name or len(well_name) < 2:
        raise ValueError(
            f"Invalid well name: '{well_name}'. Expected format like 'B5', 'A1'"
        )

    if not well_name[0].isalpha():
        raise ValueError(
            f"Invalid well name: '{well_name}'. First character must be a letter"
        )

    if not well_name[1:].isdigit():
        raise ValueError(
            f"Invalid well name: '{well_name}'. Expected format like 'B5', 'A1' "
            f"(letter followed by number)"
        )

    row = ord(well_name[0].upper()) - ord("A")
    col = int(well_name[1:]) - 1
    return row, col


def load_plate_map(path: Path) -> dict[str, dict[str, str]]:
    """Load plate map from JSON file.

    Parameters
    ----------
    path : Path
        Path to plate map JSON file

    Returns
    -------
    dict[str, dict[str, str]]
        Dictionary mapping well names to condition info
    """
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    plate_map = {}
    for well_data in data:
        well_name = well_data[0]
        condition_name, color = well_data[2]
        plate_map[well_name] = {"name": condition_name, "color": color}

    return plate_map


def roi_from_roi_data(
    roi_data: ROIData,
    fov_id: int | None,
    label_value: int,
    settings_id: int | None = None,
) -> tuple[ROI, Traces, DataAnalysis, Mask, Mask | None]:
    """Convert ROIData dataclass to SQLModel entities.

    This helper function converts the existing ROIData dataclass format
    to the new normalized SQLModel format with separate tables for different
    data types. The relationships between entities will be established
    automatically via SQLModel after the ROI is added to the session.

    Parameters
    ----------
    roi_data : ROIData
        Original ROIData from analysis
    fov_id : int | None
        Parent FOV database ID (can be None before saving to DB)
    label_value : int
        ROI label number
    settings_id : int | None
        Analysis settings ID to associate with this ROI

    Returns
    -------
    tuple[ROI, Traces, DataAnalysis, Mask, Mask | None]
        Tuple of SQLModel instances ready to be added to database:
        (roi, trace, data_analysis, roi_mask, neuropil_mask)

        Note: Add masks first to get their IDs, then set the foreign keys on ROI.

    Example
    -------
    >>> from cali._plate_viewer._util import ROIData
    >>> roi_data = ROIData(...)  # from existing analysis
    >>> roi, trace, data_analysis, roi_mask, neuropil_mask = roi_from_roi_data(
    ...     roi_data, fov_id=1, label_value=1
    ... )
    >>> # Add masks first to get their IDs
    >>> session.add(roi_mask)
    >>> if neuropil_mask:
    ...     session.add(neuropil_mask)
    >>> session.flush()
    >>>
    >>> # Set mask foreign keys on ROI
    >>> roi.roi_mask_id = roi_mask.id
    >>> if neuropil_mask:
    ...     roi.neuropil_mask_id = neuropil_mask.id
    >>>
    >>> # Now add ROI and related entities
    >>> session.add(roi)
    >>> session.flush()
    >>> trace.roi_id = roi.id
    >>> data_analysis.roi_id = roi.id
    >>> session.add_all([trace, data_analysis])
    >>> session.commit()
    """
    # Create ROI core (fov_id placeholder if None, will be set via relationship)
    roi = ROI(
        fov_id=fov_id if fov_id is not None else 0,
        label_value=label_value,
        analysis_settings_id=settings_id,
        active=roi_data.active,
        stimulated=roi_data.stimulated,
    )

    # Create Trace (roi_id will be set after ROI is added to session)
    trace = Traces(
        raw_trace=roi_data.raw_trace,
        corrected_trace=roi_data.corrected_trace,
        neuropil_trace=roi_data.neuropil_trace,
        dff=roi_data.dff,
        dec_dff=roi_data.dec_dff,
        x_axis=roi_data.elapsed_time_list_ms,
        roi=roi,  # Use relationship instead
    )

    # Create DataAnalysis
    data_analysis = DataAnalysis(
        cell_size=roi_data.cell_size,
        cell_size_units=roi_data.cell_size_units,
        total_recording_time_sec=roi_data.total_recording_time_sec,
        peaks_dec_dff=roi_data.peaks_dec_dff,
        peaks_amplitudes_dec_dff=roi_data.peaks_amplitudes_dec_dff,
        dec_dff_frequency=roi_data.dec_dff_frequency,
        iei=roi_data.iei,
        inferred_spikes=roi_data.inferred_spikes,
        roi=roi,  # Use relationship instead
    )

    # Handle ROI mask coordinate conversion
    roi_mask_y, roi_mask_x, roi_mask_h, roi_mask_w = None, None, None, None
    if roi_data.mask_coord_and_shape:
        (y_coords, x_coords), (height, width) = roi_data.mask_coord_and_shape
        roi_mask_y, roi_mask_x = y_coords, x_coords
        roi_mask_h, roi_mask_w = height, width

    # Handle neuropil mask coordinate conversion
    neuropil_mask_y, neuropil_mask_x = None, None
    neuropil_mask_h, neuropil_mask_w = None, None
    if roi_data.neuropil_mask_coord_and_shape:
        (y_coords, x_coords), (height, width) = roi_data.neuropil_mask_coord_and_shape
        neuropil_mask_y, neuropil_mask_x = y_coords, x_coords
        neuropil_mask_h, neuropil_mask_w = height, width

    # Create ROI Mask
    roi_mask = Mask(
        coords_y=roi_mask_y,
        coords_x=roi_mask_x,
        height=roi_mask_h,
        width=roi_mask_w,
        mask_type="roi",
    )

    # Create Neuropil Mask (if data available)
    neuropil_mask = None
    if roi_data.neuropil_mask_coord_and_shape:
        neuropil_mask = Mask(
            coords_y=neuropil_mask_y,
            coords_x=neuropil_mask_x,
            height=neuropil_mask_h,
            width=neuropil_mask_w,
            mask_type="neuropil",
        )

    return (roi, trace, data_analysis, roi_mask, neuropil_mask)


def save_experiment_to_db(
    experiment: Experiment,
    db_path: Path | str,
    overwrite: bool = False,
    keep_session: bool = False,
) -> Session | None:
    """Save an experiment object tree to a SQLite database.

    Parameters
    ----------
    experiment : Experiment
        Experiment object (e.g., from load_analysis_from_json)
    db_path : Path | str
        Path to SQLite database file
    overwrite : bool, optional
        Whether to overwrite existing database file, by default False
    keep_session : bool, optional
        Whether to return an open Session after saving, by default False

    Example
    -------
    >>> from pathlib import Path
    >>> exp = load_analysis_from_json(Path("tests/test_data/..."))
    >>> save_experiment_to_db(exp, "analysis.db")
    """
    db_path = Path(db_path)
    if overwrite and db_path.exists():
        db_path.unlink()

    engine = create_engine(f"sqlite:///{db_path}")
    create_db_and_tables(engine)

    session = Session(engine)
    session.add(experiment)
    session.commit()

    if not keep_session:
        session.close()
        session = None

    return session
