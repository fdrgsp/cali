"""Shared helpers for converting OME plate metadata to useq.WellPlatePlan.

Provides the reverse mapping of ome-writers' useq_to_acquisition_settings():
OME plate definitions -> useq.WellPlatePlan.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

import useq

if TYPE_CHECKING:
    from ome_types.model import Plate as OMEPlate


# Default well spacing/size when not available from metadata
_DEFAULT_WELL_SPACING = (9.0, 9.0)  # mm, typical 96-well
_DEFAULT_WELL_SIZE = (6.0, 6.0)  # mm, typical 96-well


def ome_plate_to_plate_plan(
    ome_plate: OMEPlate,
    *,
    fovs_per_well: int = 1,
) -> useq.WellPlatePlan:
    """Convert an ome-types Plate to a useq.WellPlatePlan.

    Parameters
    ----------
    ome_plate : ome_types.model.Plate
        OME plate metadata (from ome_types.from_tiff or similar).
    fovs_per_well : int
        Number of fields of view per well. Determined from well_samples count.

    Returns
    -------
    useq.WellPlatePlan
    """
    rows = ome_plate.rows or 1
    columns = ome_plate.columns or 1

    # Try to match a known plate by dimensions, otherwise create custom
    plate = _find_or_create_plate(rows, columns, name=ome_plate.name)

    # Build selected wells from ome plate wells
    well_rows = tuple(w.row for w in ome_plate.wells)
    well_cols = tuple(w.column for w in ome_plate.wells)
    selected_wells = (well_rows, well_cols) if ome_plate.wells else None

    well_points_plan = useq.RandomPoints(num_points=fovs_per_well)

    return useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0.0, 0.0),
        selected_wells=selected_wells,
        well_points_plan=well_points_plan,
    )


def ngff_plate_to_plate_plan(
    plate_attrs: dict[str, Any],
    *,
    fovs_per_well: int = 1,
) -> useq.WellPlatePlan:
    """Convert NGFF plate metadata (from .zattrs) to a useq.WellPlatePlan.

    Parameters
    ----------
    plate_attrs : dict
        The "plate" dict from NGFF .zattrs. Expected keys:
        "rows", "columns", "wells" (list of {path, rowIndex, columnIndex}).
    fovs_per_well : int
        Number of fields of view per well.

    Returns
    -------
    useq.WellPlatePlan
    """
    ngff_rows = plate_attrs.get("rows", [])
    ngff_cols = plate_attrs.get("columns", [])
    rows = len(ngff_rows)
    columns = len(ngff_cols)

    plate_name = plate_attrs.get("name")
    plate = _find_or_create_plate(rows, columns, name=plate_name)

    # Build selected wells
    wells = plate_attrs.get("wells", [])
    well_rows = tuple(w["rowIndex"] for w in wells)
    well_cols = tuple(w["columnIndex"] for w in wells)
    selected_wells = (well_rows, well_cols) if wells else None

    well_points_plan = useq.RandomPoints(num_points=fovs_per_well)

    return useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0.0, 0.0),
        selected_wells=selected_wells,
        well_points_plan=well_points_plan,
    )


def build_time_plan(
    size_t: int,
    time_increment: float | None = None,
) -> useq.TIntervalLoops | None:
    """Build a useq time plan from dimension sizes."""
    if size_t <= 1:
        return None
    interval = timedelta(seconds=time_increment) if time_increment else timedelta(0)
    return useq.TIntervalLoops(interval=interval, loops=size_t)


def build_z_plan(size_z: int) -> useq.ZRangeAround | None:
    """Build a useq z plan from dimension size."""
    if size_z <= 1:
        return None
    return useq.ZRangeAround(range=size_z, step=1.0)


def build_channels(
    size_c: int,
    channel_names: list[str] | None = None,
    exposure: float = 100.0,
) -> tuple[useq.Channel, ...] | None:
    """Build useq channels from dimension size."""
    if size_c <= 1:
        return None
    names = channel_names or [f"ch{i}" for i in range(size_c)]
    return tuple(
        useq.Channel(config=name, exposure=exposure) for name in names[:size_c]
    )


def build_sequence(
    stage_positions: useq.WellPlatePlan | list[useq.Position],
    size_t: int = 1,
    size_z: int = 1,
    size_c: int = 1,
    time_increment: float | None = None,
    channel_names: list[str] | None = None,
    exposure: float = 100.0,
) -> useq.MDASequence:
    """Build an MDASequence from dimensions and positions."""
    kwargs: dict[str, Any] = {"stage_positions": stage_positions}

    time_plan = build_time_plan(size_t, time_increment)
    if time_plan:
        kwargs["time_plan"] = time_plan

    z_plan = build_z_plan(size_z)
    if z_plan:
        kwargs["z_plan"] = z_plan

    channels = build_channels(size_c, channel_names, exposure)
    if channels:
        kwargs["channels"] = channels

    return useq.MDASequence(**kwargs)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_or_create_plate(
    rows: int, columns: int, name: str | None = None
) -> useq.WellPlate:
    """Find a known plate matching dimensions, or create a custom one."""
    # Try matching known plates by row/column count
    for plate_name in useq.registered_well_plate_keys():
        known = useq.WellPlate.from_str(plate_name)
        if known.rows == rows and known.columns == columns:
            return known

    # No match - create custom plate
    return useq.WellPlate(
        rows=rows,
        columns=columns,
        well_spacing=_DEFAULT_WELL_SPACING,
        well_size=_DEFAULT_WELL_SIZE,
        name=name or f"custom-{rows}x{columns}",
    )
