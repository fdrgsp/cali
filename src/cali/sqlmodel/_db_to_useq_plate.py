"""Convert database Experiment back to useq-schema WellPlate and WellPlatePlan.

This module provides utilities to reconstruct useq.WellPlate and useq.WellPlatePlan
objects from the database representation, enabling round-trip conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import useq

if TYPE_CHECKING:
    from ._model import Experiment

from cali.logger import cali_logger


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
    >>> from cali.sqlmodel import (
    ...     load_experiment_from_database,
    ...     experiment_to_useq_plate,
    ... )
    >>>
    >>> exp = load_experiment_from_database("analysis.db")
    >>> if exp:
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
    >>> from cali.sqlmodel import (
    ...     load_experiment_from_database,
    ...     experiment_to_useq_plate_plan,
    ... )
    >>>
    >>> exp = load_experiment_from_database("analysis.db")
    >>> if exp:
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

    # Build explicit paired coordinates (zero-indexed!)
    # Format: ((row1, row2, ...), (col1, col2, ...))
    # Sort wells to ensure consistent order
    wells_sorted = sorted(experiment.plate.wells, key=lambda w: (w.row, w.column))
    rows = tuple(w.row for w in wells_sorted)
    cols = tuple(w.column for w in wells_sorted)
    selected_wells = (rows, cols) if rows and cols else None

    # Create the WellPlatePlan
    try:
        return useq.WellPlatePlan(
            plate=useq_plate,
            a1_center_xy=(0.0, 0.0),
            rotation=None,
            selected_wells=selected_wells,
        )
    except Exception as e:
        cali_logger.error(f"Failed to create WellPlatePlan: {e}")
        return None
