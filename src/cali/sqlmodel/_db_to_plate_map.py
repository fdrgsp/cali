"""Convert SQLModel Experiment data to plate map format.

This module provides utilities to extract condition data from an Experiment
and convert it to the PlateMapData format used by the plate viewer GUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cali._plate_viewer._plate_map import PlateMapData
    from cali.sqlmodel._models import Experiment


def experiment_to_plate_map_data(
    experiment: Experiment,
) -> tuple[list[PlateMapData], list[PlateMapData]]:
    """Convert Experiment conditions to plate map format.

    Extracts condition_1 and condition_2 from each well in the experiment
    and converts them to PlateMapData format that can be loaded into the
    plate map widget. Note that condition_1 and condition_2 refer to the
    first and second conditions in each well's conditions list, NOT filtered
    by condition_type.

    Parameters
    ----------
    experiment : Experiment
        The experiment with plate, wells, and conditions to convert

    Returns
    -------
    tuple[list[PlateMapData], list[PlateMapData]]
        A tuple of (condition_1_data, condition_2_data) where each list
        contains PlateMapData objects for wells with that condition type.
        Empty lists are returned if no conditions are found.

    Examples
    --------
    >>> experiment = load_experiment_from_db(db_path, "my_experiment")
    >>> cond1_data, cond2_data = experiment_to_plate_map_data(experiment)
    >>> # Use with _load_plate_map or plate_map_wdg.setValue()
    >>> plate_map_wdg.setValue(cond1_data, cond2_data)
    """
    from cali._plate_viewer._plate_map import PlateMapData

    condition_1_data: list[PlateMapData] = []
    condition_2_data: list[PlateMapData] = []

    # If no plate, return empty lists
    if not experiment.plate or not experiment.plate.wells:
        return condition_1_data, condition_2_data

    # Iterate through all wells in the plate
    for well in experiment.plate.wells:
        # Get condition_1 (first condition in the list)
        # Note: Using the condition_1 property which returns the first condition
        # in the well's conditions list, regardless of type
        if well.condition_1:
            plate_map_entry = PlateMapData(
                name=well.name,
                row_col=(well.row, well.column),
                condition=(
                    well.condition_1.name,
                    well.condition_1.color or "gray",  # Default color if none
                ),
            )
            condition_1_data.append(plate_map_entry)

        # Get condition_2 (second condition in the list)
        # Note: Using the condition_2 property which returns the second condition
        # in the well's conditions list, regardless of type
        if well.condition_2:
            plate_map_entry = PlateMapData(
                name=well.name,
                row_col=(well.row, well.column),
                condition=(
                    well.condition_2.name,
                    well.condition_2.color or "gray",  # Default color if none
                ),
            )
            condition_2_data.append(plate_map_entry)

    return condition_1_data, condition_2_data
