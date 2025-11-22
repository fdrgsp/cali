"""Convert useq-schema WellPlatePlan to cali.sqlmodel Plate and Wells.

This module provides utilities to create cali.sqlmodel database objects from
useq-schema plate definitions, enabling easy database initialization from
experimental plate layouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._model import Condition, Plate

if TYPE_CHECKING:
    import useq

    from ._model import Experiment

from cali.logger import cali_logger


def _row_index_to_label(row: int) -> str:
    """Convert 0-based row index to well plate row label (A, B, ..., Z, AA, AB, ...).

    Parameters
    ----------
    row : int
        0-based row index (0=A, 1=B, ..., 25=Z, 26=AA, 27=AB, ...)

    Returns
    -------
    str
        Row label (e.g., 'A', 'Z', 'AA', 'AE')

    Examples
    --------
    >>> _row_index_to_label(0)
    'A'
    >>> _row_index_to_label(25)
    'Z'
    >>> _row_index_to_label(26)
    'AA'
    >>> _row_index_to_label(30)
    'AE'
    """
    label = ""
    row += 1  # Convert to 1-based for easier calculation
    while row > 0:
        row -= 1
        label = chr(ord("A") + (row % 26)) + label
        row //= 26
    return label


def _row_label_to_index(label: str) -> int:
    """Convert row label to 0-based index (A→0, B→1, ..., Z→25, AA→26).

    Parameters
    ----------
    label : str
        Row label (e.g., 'A', 'Z', 'AA', 'AE')

    Returns
    -------
    int
        0-based row index (0=A, 1=B, ..., 25=Z, 26=AA, 27=AB, ...)

    Examples
    --------
    >>> _row_label_to_index("A")
    0
    >>> _row_label_to_index("Z")
    25
    >>> _row_label_to_index("AA")
    26
    >>> _row_label_to_index("AE")
    30
    """
    index = 0
    for char in label.upper():
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1  # Convert to 0-based


def useq_plate_to_db(
    useq_plate: useq.WellPlate,
    experiment: Experiment,
    plate_maps: dict[str, dict[str, str]] | None = None,
) -> Plate:
    """Convert useq.WellPlate to cali.sqlmodel Plate (without Wells).

    This creates a Plate object with basic plate metadata but no Wells.
    Use this when you want to create a plate structure and add wells later,
    or use `useq_plate_plan_to_db` to create both plate and wells together.

    Parameters
    ----------
    useq_plate : useq.WellPlate
        useq-schema WellPlate containing plate definition
    experiment : Experiment
        Parent experiment object to associate with the plate
    plate_maps : dict[str, dict[str, str]] | None
        Optional plate map configuration mapping well positions to conditions.
        Format: {"genotype": {"A1": "WT", "A2": "KO", ...},
                 "treatment": {"A1": "Vehicle", "A2": "Drug", ...}}
        This will be stored on the plate but wells won't be created by this function.

    Returns
    -------
    Plate
        Plate object without Wells. If plate_maps provided, it will be stored
        on plate.plate_maps.

    Example
    -------
    >>> import useq
    >>> from cali.sqlmodel import Experiment, useq_plate_to_db
    >>>
    >>> # Create experiment
    >>> exp = Experiment(name="my_experiment", description="Test")
    >>>
    >>> # Create plate with plate_maps
    >>> useq_plate = useq.WellPlate.from_str("96-well")
    >>> plate_maps = {"genotype": {"A1": "WT", "A2": "KO"}}
    >>> plate = useq_plate_to_db(useq_plate, exp, plate_maps=plate_maps)
    >>> print(f"Created plate '{plate.name}' with {plate.rows}x{plate.columns} layout")
    Created plate '96-well' with 8x12 layout
    """
    if experiment.id is None:
        raise ValueError("Experiment must have an ID before creating a Plate.")

    # this is because the 18 and 22 mm coverslips name are different from the name that
    # useq uses to create the plate and in cali we use the plate.plate_type to
    # eventually reconstruct the useq plate.
    if useq_plate.name == "18mm coverslip":
        plate_type = "coverslip-18mm-square"
    elif useq_plate.name == "22mm coverslip":
        plate_type = "coverslip-22mm-square"
    else:
        plate_type = useq_plate.name
    plate = Plate(
        experiment_id=experiment.id,
        experiment=experiment,
        name=useq_plate.name,
        plate_type=plate_type,
        rows=useq_plate.rows,
        columns=useq_plate.columns,
        plate_maps=plate_maps,
    )

    return plate


def useq_plate_plan_to_db(
    plate_plan: useq.WellPlatePlan,
    experiment: Experiment,
    plate_maps: dict[str, dict[str, str]] | None = None,
) -> Plate:
    """Convert useq.WellPlatePlan to cali.sqlmodel Plate with Wells and FOVs.

    This creates a Plate object and populates it with Well and FOV objects based on
    the positions in the WellPlatePlan. The Wells will have proper row/column
    indices and FOVs with position information extracted from the plan.

    Optionally, provide plate_maps to automatically create and assign conditions
    to wells based on well positions.

    Parameters
    ----------
    plate_plan : useq.WellPlatePlan
        useq-schema WellPlatePlan containing plate definition and positions
    experiment : Experiment
        Parent experiment object to associate with the plate
    plate_maps : dict[str, dict[str, str]] | None
        Optional plate map configuration mapping well positions to conditions.
        Format: {"genotype": {"A1": "WT", "A2": "KO", ...},
                 "treatment": {"A1": "Vehicle", "A2": "Drug", ...}}
        If provided, conditions will be created and assigned to wells.

    Returns
    -------
    Plate
        Plate object with Wells and FOVs created for all positions in the plan.
        If plate_maps provided, the plate.plate_maps field will be set and
        wells will have conditions assigned.

    Example
    -------
    >>> import useq
    >>> from cali.sqlmodel import Experiment, useq_plate_plan_to_db
    >>>
    >>> # Create experiment
    >>> exp = Experiment(name="my_experiment", description="Test")
    >>>
    >>> # Create plate plan (typically loaded from data metadata)
    >>> plate_plan = useq.WellPlatePlan(
    ...     plate=useq.WellPlate.from_str("96-well"),
    ...     selected_wells=((1, 4),),  # Well B5
    ... )
    >>>
    >>> # Convert to database objects with conditions
    >>> plate_maps = {
    ...     "genotype": {"B5": "WT"},
    ...     "treatment": {"B5": "Vehicle"},
    ... }
    >>> plate = useq_plate_plan_to_db(plate_plan, exp, plate_maps=plate_maps)
    >>> print(f"Created plate '{plate.name}' with {len(plate.wells)} wells")
    Created plate '96-well' with 1 wells
    >>> print(f"Well B5 FOVs: {len(plate.wells[0].fovs)}")
    Well B5 FOVs: 2
    >>> print(f"Well B5 conditions: {[c.name for c in plate.wells[0].conditions]}")
    Well B5 conditions: ['WT', 'Vehicle']
    """
    from collections import defaultdict

    from ._model import FOV, Well

    useq_plate = plate_plan.plate

    # Create the Plate object using the helper function with plate_maps
    plate = useq_plate_to_db(useq_plate, experiment, plate_maps=plate_maps)

    # Create a condition cache to avoid duplicates
    condition_cache: dict[tuple[str, str], Condition] = {}

    # Helper function to get or create condition
    def get_or_create_condition(cond_name: str, cond_type: str) -> Condition:
        key = (cond_type, cond_name)
        if key not in condition_cache:
            condition_cache[key] = Condition(
                name=cond_name,
                condition_type=cond_type,
            )
        return condition_cache[key]

    # Collect positions by well from the plate plan
    # Position names are like "B5_0000" where B5 is well and 0000 is FOV index
    wells_data: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for pos in plate_plan:
        if pos.name:
            # Parse name like "B5_0000" -> well="B5", fov_index=0
            parts = pos.name.rsplit("_", 1)
            if len(parts) == 2:
                well_name, fov_str = parts
                try:
                    fov_index = int(fov_str)
                    wells_data[well_name].append(
                        (fov_index, pos.x or 0.0, pos.y or 0.0)
                    )
                except ValueError:
                    # If we can't parse FOV index, skip this position
                    cali_logger.warning(
                        f"Invalid FOV index in position name '{pos.name}', skipping."
                    )
                    continue
            else:
                # single fov per well, no index
                well_name, fov_index = parts[0], 0
                wells_data[well_name].append((0, pos.x or 0.0, pos.y or 0.0))

    # Create Wells and FOVs
    for well_name, fov_data in sorted(wells_data.items()):
        # Parse well name to get row and column
        # Well names like "B5" -> row=1 (B), col=4 (5-1)
        row_label = "".join(c for c in well_name if c.isalpha())
        col_str = "".join(c for c in well_name if c.isdigit())

        if not row_label or not col_str:
            continue

        # Convert row label to index (A=0, B=1, ..., AA=26, etc.)
        row = _row_label_to_index(row_label)
        col = int(col_str) - 1  # Column is 0-indexed internally

        # Collect conditions for this well from plate_maps
        well_conditions = []
        if plate_maps:
            for condition_type, well_map in plate_maps.items():
                if well_name in well_map:
                    condition_name = well_map[well_name]
                    condition = get_or_create_condition(condition_name, condition_type)
                    well_conditions.append(condition)

        # Create well
        well = Well(
            plate_id=0,  # placeholder, will be set by relationship
            plate=plate,
            name=well_name,
            row=row,
            column=col,
            conditions=well_conditions,
        )

        # Create FOVs for this well
        # Sort by fov_index to ensure consistent ordering
        for fov_index, _, _ in sorted(fov_data):
            _ = FOV(
                well_id=0,  # placeholder, will be set by relationship
                well=well,
                name=f"{well_name}_{fov_index:04d}",
                position_index=fov_index,
                fov_number=fov_index,
            )

    return plate
