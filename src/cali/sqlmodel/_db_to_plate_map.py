"""Convert SQLModel Experiment data to plate map format.

This module provides utilities to extract condition data from an Experiment
and convert it to the PlateMapData format used by the plate viewer GUI.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cali.gui._plate_map import PlateMapData
    from cali.sqlmodel._model import Experiment


def _generate_random_color() -> str:
    """Generate a random color in hex format.

    Returns
    -------
    str
        A random color in hex format (e.g., "#3A7F9B")
    """
    return f"#{random.randint(0, 0xFFFFFF):06X}"


def experiment_to_plate_map_data(
    experiment: Experiment,
) -> tuple[list[PlateMapData], list[PlateMapData]]:
    """Convert Experiment conditions to plate map format.

    Extracts genotype and treatment conditions from each well in the experiment
    and converts them to PlateMapData format that can be loaded into the
    plate map widget. Conditions are grouped by their condition_type, ensuring
    that genotype conditions are always returned first and treatment conditions
    second, regardless of their order in the well's conditions list.

    Parameters
    ----------
    experiment : Experiment
        The experiment with plate, wells, and conditions to convert

    Returns
    -------
    tuple[list[PlateMapData], list[PlateMapData]]
        A tuple of (genotype_data, treatment_data) where each list
        contains PlateMapData objects for wells with that condition type.
        Empty lists are returned if no conditions are found.

    Examples
    --------
    >>> from cali.sqlmodel import load_experiment_from_database
    >>> experiment = load_experiment_from_database("analysis.db", "my_experiment")
    >>> genotype_data, treatment_data = experiment_to_plate_map_data(experiment)
    >>> # Use with _load_plate_map or plate_map_wdg.setValue()
    >>> plate_map_wdg.setValue(genotype_data, treatment_data)
    """
    from cali.gui._plate_map import PlateMapData

    genotype_data: list[PlateMapData] = []
    treatment_data: list[PlateMapData] = []

    # If no plate, return empty lists
    if not experiment.plate or not experiment.plate.wells:
        return genotype_data, treatment_data

    # Iterate through all wells in the plate
    for well in experiment.plate.wells:
        # Look for genotype and treatment conditions explicitly by type
        for condition in well.conditions:
            if condition.condition_type == "genotype":
                plate_map_entry = PlateMapData(
                    name=well.name,
                    row_col=(well.row, well.column),
                    condition=(
                        condition.name,
                        condition.color or _generate_random_color(),
                    ),
                )
                genotype_data.append(plate_map_entry)
            elif condition.condition_type == "treatment":
                plate_map_entry = PlateMapData(
                    name=well.name,
                    row_col=(well.row, well.column),
                    condition=(
                        condition.name,
                        condition.color or _generate_random_color(),
                    ),
                )
                treatment_data.append(plate_map_entry)

    return genotype_data, treatment_data
