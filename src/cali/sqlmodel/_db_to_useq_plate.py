"""Convert database Experiment back to useq-schema WellPlate and WellPlatePlan.

This module provides utilities to reconstruct useq.WellPlate and useq.WellPlatePlan
objects from the database representation, enabling round-trip conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import useq

if TYPE_CHECKING:
    from ._models import Experiment


def experiment_to_useq_plate(
    experiment: Experiment, useq_plate_name: str | None = None
) -> useq.WellPlate | None:
    """Convert database Experiment to useq.WellPlate.

    This reconstructs a useq.WellPlate from the database representation,
    extracting plate type, dimensions, and well information.

    Parameters
    ----------
    experiment : Experiment
        Database experiment with plate and well information
    useq_plate_name : str | None
        Optional name for the reconstructed WellPlate. If None, uses plate_type
        from the database.

    Returns
    -------
    useq.WellPlate | None
        Reconstructed useq-schema WellPlate object or None if plate type is not
        specified in the Experiment.plate.

    Example
    -------
    >>> from sqlmodel import create_engine, Session, select
    >>> from cali.sqlmodel import Experiment, experiment_to_useq_plate
    >>>
    >>> engine = create_engine("sqlite:///analysis.db")
    >>> with Session(engine) as session:
    ...     exp = session.exec(select(Experiment)).first()
    ...     plate = experiment_to_useq_plate(exp)
    ...     print(f"Plate: {plate.name}, Wells: {len(exp.plate.wells)}")
    """
    plate = experiment.plate

    if useq_plate_name:
        try:
            return useq.WellPlate.from_str(useq_plate_name)
        except Exception as e:
            raise ValueError(
                f"Invalid useq.WellPlate name. Cannot find '{useq_plate_name}' in useq"
                f" registered plates: {e}."
            ) from e

    # Determine plate type from the stored plate_type
    if plate.plate_type:
        # Try to create from standard name (e.g., "96-well", "384-well")
        try:
            return useq.WellPlate.from_str(plate.plate_type)
        except Exception as e:
            raise ValueError(
                f"Invalid useq.WellPlate name. Cannot find '{plate.plate_type}' in useq"
                f" registered plates: {e}."
            ) from e

    return None


def experiment_to_useq_plate_plan(
    experiment: Experiment,
    useq_plate_name: str | None = None,
) -> useq.WellPlatePlan | None:
    """Convert database Experiment to useq.WellPlatePlan.

    This reconstructs a useq.WellPlatePlan from the database representation,
    extracting plate information and determining which wells were imaged based
    on the wells present in the database.

    Note that the a1_center_xy and rotation are set to default values, as this
    information is not used internally in cali.

    Parameters
    ----------
    experiment : Experiment
        Database experiment with plate and well information
    useq_plate_name : str | None, optional
        Optional name for the reconstructed WellPlate. If None, uses the plate_type
        from the database.

    Returns
    -------
    useq.WellPlatePlan | None
        Reconstructed useq-schema WellPlatePlan object with the plate definition
        and the wells that were imaged, or None if plate type is not specified.

    Notes
    -----
    The selected_wells are automatically determined from the wells present in the
    database. If you had wells B5, B6, C5, C6 in the database, the selected_wells
    will be set to ((1, 2), (4, 5)) which represents rows B-C and columns 5-6.

    Example
    -------
    >>> from sqlmodel import create_engine, Session, select
    >>> from cali.sqlmodel import Experiment, experiment_to_useq_plate_plan
    >>>
    >>> engine = create_engine("sqlite:///analysis.db")
    >>> with Session(engine) as session:
    ...     exp = session.exec(select(Experiment)).first()
    ...     plate_plan = experiment_to_useq_plate_plan(exp)
    ...     print(f"Plate: {plate_plan.plate.name}")
    ...     print(f"Selected wells (rows, cols): {plate_plan.selected_wells}")
    """
    # Get the plate object
    useq_plate = experiment_to_useq_plate(experiment, useq_plate_name)
    if useq_plate is None:
        return None

    # Extract wells from the experiment
    if not experiment.plate or not experiment.plate.wells:
        # No wells in database, return None
        return None

    # Determine selected wells from the wells present in the database
    rows = set()
    cols = set()
    for well in experiment.plate.wells:
        rows.add(well.row)
        cols.add(well.column)

    # Convert to sorted tuples (required format for useq.WellPlatePlan)
    selected_rows = tuple(sorted(rows))
    selected_cols = tuple(sorted(cols))
    selected_wells = (selected_rows, selected_cols)

    # Create the WellPlatePlan
    return useq.WellPlatePlan(
        plate=useq_plate,
        a1_center_xy=(0.0, 0.0),
        rotation=None,
        selected_wells=selected_wells,
    )
