"""Convert useq-schema WellPlatePlan to cali.sqlmodel Plate and Wells.

This module provides utilities to create cali.sqlmodel database objects from
useq-schema plate definitions, enabling easy database initialization from
experimental plate layouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import useq

    from ._models import Experiment, Plate


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


def useq_plate_plan_to_plate(
    plate_plan: useq.WellPlatePlan,
    experiment: Experiment,
) -> Plate:
    """Convert useq.WellPlatePlan to cali.sqlmodel Plate with Wells.

    This creates a Plate object and populates it with Well objects based on
    the selected wells in the WellPlatePlan. The Wells will have proper row/column
    indices but no FOVs or analysis data (those are added during analysis).

    Parameters
    ----------
    plate_plan : useq.WellPlatePlan
        useq-schema WellPlatePlan containing plate definition and selected wells
    experiment : Experiment
        Parent experiment object to associate with the plate

    Returns
    -------
    Plate
        Plate object with Wells created for all selected wells in the plan

    Example
    -------
    >>> import useq
    >>> from cali.sqlmodel import Experiment, useq_plate_plan_to_plate
    >>>
    >>> # Create experiment
    >>> exp = Experiment(name="my_experiment", description="Test")
    >>>
    >>> # Create plate plan
    >>> plate_plan = useq.WellPlatePlan(
    ...     plate=useq.WellPlate.from_str("96-well"),
    ...     selected_wells=((1, 2), (4, 5)),  # Wells B5, C6 (paired indices)
    ... )
    >>>
    >>> # Convert to database objects
    >>> plate = useq_plate_plan_to_plate(plate_plan, exp)
    >>> print(f"Created plate '{plate.name}' with {len(plate.wells)} wells")
    Created plate '96-well' with 2 wells
    """
    from ._models import Plate, Well

    useq_plate = plate_plan.plate

    # Create the Plate object
    plate = Plate(
        experiment_id=0,  # Placeholder, will be set when saved to DB
        experiment=experiment,
        name=useq_plate.name,
        plate_type=useq_plate.name,
        rows=useq_plate.rows,
        columns=useq_plate.columns,
    )

    # Create Wells for selected wells
    if plate_plan.selected_wells:
        selected_rows, selected_cols = plate_plan.selected_wells

        # selected_wells uses paired indices, not a grid
        # E.g., ((1, 2), (4, 5)) means wells (1,4) and (2,5), not all 4 combinations
        for row, col in zip(selected_rows, selected_cols, strict=True):
            # Convert row index to letter (0=A, 1=B, ..., 26=AA, 30=AE, etc.)
            row_label = _row_index_to_label(row)
            # Column is 1-indexed in well names
            well_name = f"{row_label}{col + 1}"

            # Create well - it's automatically added to plate.wells via relationship
            _ = Well(
                plate_id=0,  # Placeholder, will be set when saved to DB
                plate=plate,
                name=well_name,
                row=row,
                column=col,
                conditions=[],  # Conditions added separately if needed
            )

    return plate
