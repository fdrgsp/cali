from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import useq

from cali.logger import cali_logger
from cali.sqlmodel._useq_plate_to_db import useq_plate_plan_to_db
from cali.util import load_data

if TYPE_CHECKING:
    from cali.readers import OMEZarrReader, TensorstoreZarrReader
    from cali.sqlmodel import Experiment, Plate


def data_to_plate(
    data: str | Path | TensorstoreZarrReader | OMEZarrReader,
    experiment: Experiment,
    plate_maps: dict[str, dict[str, str]] | None = None,
) -> Plate | None:
    if isinstance(data, (str, Path)):
        dataset = load_data(data)
        if dataset is None:
            cali_logger.error(f"Could not load data from path: {data}")
            return None
    else:
        dataset = data

    if dataset.sequence is None:
        cali_logger.error("Dataset does not contain sequence information.")
        return None

    plate_plan = dataset.sequence.stage_positions
    if not isinstance(plate_plan, useq.WellPlatePlan):
        cali_logger.error("Dataset does not contain a WellPlatePlan.")
        return None

    return useq_plate_plan_to_db(plate_plan, experiment, plate_maps)
