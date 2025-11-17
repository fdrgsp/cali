"""Migration helper: Import existing JSON analysis data into SQLModel database.

This module provides functions to migrate calcium imaging analysis data
from JSON format to the new SQLModel database format.

Example
-------
>>> from pathlib import Path
>>> from cali.sqlmodel import load_analysis_from_json
>>> import useq
>>> data_dir = Path("tests/test_data/spontaneous/spont.tensorstore.zarr")
>>> analysis_dir = Path("tests/test_data/spontaneous/spont_analysis")
>>> plate = useq.WellPlate.from_str("96-well")
>>> experiment = load_analysis_from_json(
...     str(data_dir), str(analysis_dir), plate
... )
>>> print(f"Loaded {len(experiment.plate.wells)} wells")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from cali._constants import EVOKED, SPONTANEOUS

from ._model import (
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
)
from ._util import ROIData

if TYPE_CHECKING:
    from useq import WellPlate


def load_analysis_from_json(
    data_path: str,
    analysis_path: str,
    useq_plate: WellPlate,
) -> Experiment:
    """Load analysis data from JSON directory into SQLModel objects.

    This function reads all JSON files in the analysis directory and creates
    a complete Experiment object with all related entities (Plate, Wells,
    FOVs, ROIs, Conditions, AnalysisSettings, Masks, Traces, DataAnalysis).

    If 'genotype_plate_map.json' and/or 'treatment_plate_map.json' files exist
    in the analysis directory, they will be loaded and stored in plate.plate_maps.

    The function does NOT save to a database - it returns an in-memory object
    tree that can be added to a SQLModel session.

    Parameters
    ----------
    data_path : str
        Path to the data directory (e.g. .tensorstore.zarr)
    analysis_path : str
        Path to the analysis directory. May contain optional plate map files:
        - genotype_plate_map.json
        - treatment_plate_map.json
    useq_plate : WellPlate
        useq-schema WellPlate object defining the plate used

    Returns
    -------
    Experiment
        Complete experiment object with all relationships populated.
        The plate.plate_maps field will be populated if plate map JSON files
        were found.

    Example
    -------
    >>> from pathlib import Path
    >>> import useq
    >>> from cali.sqlmodel import load_analysis_from_json, save_experiment_to_database
    >>>
    >>> # Load from JSON
    >>> data_dir = "tests/test_data/spontaneous/spont.tensorstore.zarr"
    >>> analysis_dir = "tests/test_data/spontaneous/spont_analysis"
    >>> plate = useq.WellPlate.from_str("96-well")
    >>> experiment = load_analysis_from_json(
    ...     data_dir, analysis_dir, useq_plate=plate
    ... )
    >>> # Check if plate maps were loaded
    >>> if experiment.plate.plate_maps:
    ...     print(f"Loaded plate maps: {list(experiment.plate.plate_maps.keys())}")
    >>>
    >>> # Save to database
    >>> save_experiment_to_database(experiment, "test.db")
    """
    # 1. Create experiment
    db_name = f"{Path(data_path).name}.db"
    experiment = Experiment(
        id=0,  # placeholder, will be set when saved. Needed for relationships.
        name=db_name,
        description=f"Imported from {analysis_path}",
        data_path=data_path,
        analysis_path=analysis_path,
        database_name=db_name,
    )
    assert experiment.id is not None

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
    genotype_map_path = Path(analysis_path) / "genotype_plate_map.json"
    treatment_map_path = Path(analysis_path) / "treatment_plate_map.json"

    genotype_map = load_plate_map(genotype_map_path)
    treatment_map = load_plate_map(treatment_map_path)

    # Build plate_maps from loaded JSON files
    plate_maps: dict[str, dict[str, str]] = {}
    if genotype_map:
        plate_maps["genotype"] = {
            well_name: well_data["name"]
            for well_name, well_data in genotype_map.items()
        }
    if treatment_map:
        plate_maps["treatment"] = {
            well_name: well_data["name"]
            for well_name, well_data in treatment_map.items()
        }

    # Set plate_maps on plate if any were found
    if plate_maps:
        plate.plate_maps = plate_maps

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
    settings_path = Path(analysis_path) / "settings.json"
    analysis_settings = None

    if settings_path.exists():
        with open(settings_path) as f:
            settings_data = json.load(f)

        # Check for stimulation mask
        stimulation_mask = None
        stimulation_mask_path = None
        stim_mask_file = Path(analysis_path) / "stimulation_mask.tif"

        experiment.experiment_type = EVOKED if stim_mask_file.exists() else SPONTANEOUS

        if stim_mask_file.exists():
            try:
                # Load the stimulation mask
                import tifffile

                # Import here to avoid circular dependency
                from cali.util import mask_to_coordinates

                stim_mask_array = tifffile.imread(str(stim_mask_file))

                # Convert to coordinates for storage
                coords, shape = mask_to_coordinates(stim_mask_array.astype(bool))

                # Create Mask object
                stimulation_mask = Mask(
                    coords_y=coords[0],
                    coords_x=coords[1],
                    height=shape[0],
                    width=shape[1],
                    mask_type="stimulation",
                )

                stimulation_mask_path = str(stim_mask_file)
            except Exception as e:
                # If loading fails, just skip it (optional field)
                print(f"Warning: Could not load stimulation mask: {e}")

        analysis_settings = AnalysisSettings(
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
            stimulation_mask=stimulation_mask,
            stimulation_mask_path=stimulation_mask_path,
        )

    # 5. Process JSON files
    json_files = [
        f
        for f in Path(analysis_path).glob("*.json")
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
    positions_analyzed: set[int] = set()  # Track all unique position indices

    # Variables to collect LED stimulation info from ROI data
    led_pulse_duration_from_roi: float | None = None
    led_pulse_powers_from_roi: list[float] | None = None
    led_pulse_on_frames_from_roi: list[int] | None = None

    for json_file in json_files:
        # Parse FOV name
        fov_name = json_file.stem
        well_name = fov_name.split("_")[0]

        # Create or get well
        if well_name not in wells_created:
            row, col = parse_well_name(well_name)

            # Create well with conditions
            well = Well(
                plate_id=0,  # placeholder, will be set when saved.
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

        # Track this position as analyzed
        positions_analyzed.add(position_index)

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

                # Extract LED stimulation info from first ROI that has it
                if led_pulse_duration_from_roi is None and roi_data.led_pulse_duration:
                    led_pulse_duration_from_roi = float(roi_data.led_pulse_duration)

                if roi_data.stimulations_frames_and_powers is not None:
                    # Extract powers and frames from dict
                    # Format: {frame: power, ...}
                    stim_dict = roi_data.stimulations_frames_and_powers
                    if led_pulse_powers_from_roi is None:
                        led_pulse_powers_from_roi = [
                            float(v) for v in stim_dict.values()
                        ]
                    if led_pulse_on_frames_from_roi is None:
                        led_pulse_on_frames_from_roi = [
                            int(k) for k in stim_dict.keys()
                        ]

                # Use roi_from_roi_data helper
                roi, trace, data_analysis, roi_mask, neuropil_mask = roi_from_roi_data(
                    roi_data,
                    fov_id=fov.id or 0,  # Placeholder, will be set via relationship
                    label_value=int(roi_label),
                    settings_id=(analysis_settings.id if analysis_settings else None),
                )

                # Since we're not in DB session, manually set relationships
                roi.fov = fov  # This automatically adds roi to fov.rois
                roi.analysis_settings = analysis_settings
                roi.roi_mask = roi_mask
                roi.neuropil_mask = neuropil_mask
                roi.traces = trace
                roi.data_analysis = data_analysis

                # Note: No need to append to fov.rois - back_populates handles it

                roi_count += 1
                total_rois += 1
            except Exception as e:
                print(f"  ⚠ Error importing ROI {roi_label} from ")
                print(f"    {json_file.name}: {e}")
                continue

    # Set the positions that were analyzed
    experiment.positions_analyzed = sorted(positions_analyzed)

    # Assign analysis_settings to experiment (set relationship)
    if analysis_settings:
        experiment.analysis_settings = analysis_settings

    # Update AnalysisSettings with LED info collected from ROI data (if available)
    if analysis_settings and experiment.experiment_type == EVOKED:
        if led_pulse_duration_from_roi is not None:
            analysis_settings.led_pulse_duration = led_pulse_duration_from_roi
        if led_pulse_powers_from_roi is not None:
            analysis_settings.led_pulse_powers = led_pulse_powers_from_roi
        if led_pulse_on_frames_from_roi is not None:
            analysis_settings.led_pulse_on_frames = led_pulse_on_frames_from_roi

    return experiment


def _label_to_row_index(label: str) -> int:
    """Convert well row label to zero-indexed row number.

    Supports single and multi-letter labels using base-26 alphabet.
    A=0, B=1, ..., Z=25, AA=26, AB=27, ..., AZ=51, etc.

    Parameters
    ----------
    label : str
        Row label (e.g., 'A', 'Z', 'AA', 'AE')

    Returns
    -------
    int
        Zero-indexed row number

    Examples
    --------
    >>> _label_to_row_index("A")
    0
    >>> _label_to_row_index("Z")
    25
    >>> _label_to_row_index("AA")
    26
    >>> _label_to_row_index("AE")
    30
    """
    label = label.upper()
    result = 0
    for char in label:
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def parse_well_name(well_name: str) -> tuple[int, int]:
    """Parse well name like 'B5' or 'AE19' into (row, column) indices.

    Supports both single-letter (A-Z) and multi-letter (AA, AB, ...) row names
    for plates with more than 26 rows.

    Parameters
    ----------
    well_name : str
        Well name (e.g., 'B5', 'A1', 'AE19')

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
            f"Invalid well name: '{well_name}'. Expected format like 'B5', 'AE19'"
        )

    # Split into letter prefix and number suffix
    i = 0
    while i < len(well_name) and well_name[i].isalpha():
        i += 1

    if i == 0:
        raise ValueError(f"Invalid well name: '{well_name}'. Must start with letter(s)")

    if i == len(well_name) or not well_name[i:].isdigit():
        raise ValueError(
            f"Invalid well name: '{well_name}'. Expected format like 'B5', 'AE19' "
            f"(letter(s) followed by number)"
        )

    row_label = well_name[:i]
    row = _label_to_row_index(row_label)
    col = int(well_name[i:]) - 1
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
    >>> from cali.sqlmodel._util import ROIData
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
        # ROI-specific thresholds calculated during analysis
        peaks_prominence_dec_dff=roi_data.peaks_prominence_dec_dff,
        peaks_height_dec_dff=roi_data.peaks_height_dec_dff,
        inferred_spikes_threshold=roi_data.inferred_spikes_threshold,
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
