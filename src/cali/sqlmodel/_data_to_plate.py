from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import useq

from cali.sqlmodel._useq_plate_to_db import useq_plate_plan_to_db

if TYPE_CHECKING:
    from cali.readers import (
        OMEZarrReader,
        TensorstoreZarrReader,
        TiffCollectionReader,
    )
    from cali.sqlmodel import Experiment, Plate

from cali.logger import cali_logger


def data_to_plate(
    data: str | Path | TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
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
        if plate_plan is None:
            cali_logger.error(
                "❌  Dataset does not contain a WellPlatePlan."
                " Please provide a plate_plan to use."
            )
            return None

    return useq_plate_plan_to_db(plate_plan, experiment, plate_maps)
