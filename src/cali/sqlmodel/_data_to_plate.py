from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import useq

from cali.sqlmodel._useq_plate_to_db import useq_plate_plan_to_db

if TYPE_CHECKING:
    from collections.abc import Generator

    from cali.readers import CaliDataReader
    from cali.sqlmodel import Experiment, Plate

from cali.logger import cali_logger


def data_to_plate(
    data: str | Path | CaliDataReader,
    experiment: Experiment,
    plate_maps: dict[str, dict[str, str]] | None = None,
    plate_plan: useq.WellPlatePlan | None = None,
) -> Plate | None:
    if isinstance(data, (str, Path)):
        from cali.util import load_data_from_path

        dataset = load_data_from_path(data)
        if dataset is None:
            cali_logger.error(f"❌ Could not load data from path: {data}")
            return None
    else:
        dataset = data  # type: ignore[assignment]

    assert dataset is not None

    if dataset.sequence is None:
        cali_logger.error("❌  Dataset does not contain sequence information.")
        return None

    if isinstance(dataset.sequence.stage_positions, useq.WellPlatePlan):
        plate_plan = dataset.sequence.stage_positions
    else:
        # Data doesn't have a WellPlatePlan, so we need to use the provided one
        # and map the actual data positions to it
        if plate_plan is None:
            cali_logger.error(
                "❌  Dataset does not contain a WellPlatePlan."
                " Please provide a plate_plan to use."
            )
            return None

        # Create a modified plate plan that uses the actual data positions
        # Get the actual stage positions from the data
        stage_positions = dataset.sequence.stage_positions
        if stage_positions and len(stage_positions) > 0:
            # Get selected well names from the plate plan
            # Note: Due to a bug in pymmcore-widgets WellPlateWidget for 1x1 plates,
            # selected_wells might have duplicates. Deduplicate them.
            selected_wells = list(dict.fromkeys(plate_plan.selected_well_names))
            if len(selected_wells) == 1:
                well_name = selected_wells[0]
                # Number of FOVs per well
                num_fovs = plate_plan.num_points_per_well

                # Take the actual positions and rename them
                positions_to_use = list(stage_positions[:num_fovs])
                renamed_positions = [
                    pos.replace(name=f"{well_name}_{i:04d}")
                    for i, pos in enumerate(positions_to_use)
                ]

                # Create a new plate plan with these explicit positions
                # We use a trick: create a custom WellPlatePlan by building
                # a modified version that will iterate correctly
                from useq import MDASequence

                # Create an MDASequence with the renamed positions
                MDASequence(stage_positions=renamed_positions)

                # Create a fresh WellPlatePlan with corrected selected_wells format
                # Note: selected_wells=(0, 0) means row 0, col 0 (single well)
                # This avoids the duplicate well issue from ((0, 0),) format
                cali_logger.info(
                    f"📍 Mapping {len(renamed_positions)} data positions "
                    f"to well {well_name}"
                )

                plate_plan = useq.WellPlatePlan(
                    plate=plate_plan.plate,
                    a1_center_xy=plate_plan.a1_center_xy,
                    rotation=plate_plan.rotation,
                    selected_wells=(0, 0),  # Single well: row 0, col 0
                    well_points_plan=useq.RandomPoints(
                        num_points=len(renamed_positions)
                    ),
                )

                # Override the iteration to yield our renamed positions
                # with actual coordinates
                def custom_iter() -> Generator[useq.Position, None, None]:
                    yield from renamed_positions

                plate_plan.__iter__ = custom_iter

    return useq_plate_plan_to_db(plate_plan, experiment, plate_maps)
