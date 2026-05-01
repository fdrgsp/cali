"""Reader for OME-TIFF files.

Uses ome-types for OME-XML metadata parsing and tifffile for lazy pixel access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tifffile
import useq
from ome_types import from_tiff

from cali._constants import EVENT_KEY
from cali.readers._ome_to_useq import (
    build_sequence,
    ome_plate_to_plate_plan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from ome_types.model import OME


class OMETiffReader:
    """Reader for OME-TIFF files with lazy pixel access.

    Extracts OME-XML metadata (plate layout, pixel dimensions, channels)
    via ome-types and provides lazy data access via tifffile's zarr store.

    Parameters
    ----------
    path : str | Path
        Path to the OME-TIFF file (.ome.tif or .ome.tiff).

    Examples
    --------
    >>> reader = OMETiffReader("experiment.ome.tif")
    >>> data = reader.isel(p=0, t=5, c=0)
    >>> data, meta = reader.isel(p=0, t=5, metadata=True)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

        # Parse OME-XML metadata
        self._ome: OME = from_tiff(self._path)

        # Open TIFF for lazy access
        self._tiff = tifffile.TiffFile(self._path)

        # Extract dimensions from first image
        if not self._ome.images:
            raise ValueError(f"No images found in OME-TIFF: {self._path}")

        pixels = self._ome.images[0].pixels
        self._size_t = pixels.size_t
        self._size_c = pixels.size_c
        self._size_z = pixels.size_z
        self._size_y = pixels.size_y
        self._size_x = pixels.size_x
        self._dim_order = str(pixels.dimension_order.value)  # e.g. "XYZCT"

        # Pixel sizes for metadata
        self._pixel_size_um = pixels.physical_size_x
        self._time_increment = pixels.time_increment

        # Exposure from first plane if available
        self._exposure_ms: float | None = None
        if pixels.planes:
            exp = pixels.planes[0].exposure_time
            if exp is not None:
                self._exposure_ms = float(exp)

        # Channel names
        self._channel_names = [
            ch.name or f"ch{i}" for i, ch in enumerate(pixels.channels)
        ]

        # Build sequence
        self._plate_plan: useq.WellPlatePlan | None = None
        self._sequence = self._build_sequence()

        # Build metadata
        self._metadata_list = self._build_metadata()

    # -------------------------Properties-------------------------

    @property
    def path(self) -> Path:
        """Return the path to the OME-TIFF file."""
        return self._path

    @property
    def sequence(self) -> useq.MDASequence | None:
        """Return the MDASequence."""
        return self._sequence

    @property
    def plate_plan(self) -> useq.WellPlatePlan | None:
        """Return the WellPlatePlan if plate metadata is present."""
        return self._plate_plan

    @property
    def metadata(self) -> list[dict]:
        """Return frame-level metadata."""
        return self._metadata_list

    # -------------------------Public Methods-------------------------

    def isel(
        self,
        indexers: Mapping[str, int] | None = None,
        metadata: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, list[dict]]:
        """Select data by axis indices.

        Parameters
        ----------
        indexers : Mapping[str, int] | None
            Axis-index mapping, e.g. {"p": 0, "t": 1, "c": 0}.
        metadata : bool
            If True, also return matching metadata dicts.
        **kwargs
            Alternative way to pass indexers (e.g. p=0, t=1).
        """
        import numpy as np

        if indexers is None:
            indexers = {}
        if kwargs:
            if not all(
                isinstance(k, str) and isinstance(v, int) for k, v in kwargs.items()
            ):
                raise TypeError(
                    "kwargs must be a mapping from strings to integers (e.g. p=0, t=1)!"
                )
            indexers = {**indexers, **kwargs}

        p = indexers.get("p", 0)

        # Get the series for this position
        series = self._tiff.series
        if p >= len(series):
            raise IndexError(
                f"Position index {p} out of range (have {len(series)} series)."
            )

        # Build index tuple based on dimension order (excluding X and Y)
        idx = self._build_array_index(indexers, series[p].axes)

        # Read data lazily via zarr store
        store = series[p].aszarr()
        import zarr

        arr = zarr.open(store, mode="r")
        data = np.asarray(arr[idx]).squeeze()

        if metadata:
            meta = self._get_metadata_for_indexers(indexers)
            return data, meta
        return data

    def write_tiff(
        self,
        path: str | Path,
        indexers: Mapping[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write selected data to a TIFF file."""
        if kwargs:
            indexers = indexers or {}
            if all(
                isinstance(k, str) and isinstance(v, int) for k, v in kwargs.items()
            ):
                indexers = {**indexers, **kwargs}
            else:
                raise TypeError(
                    "kwargs must be a mapping from strings to integers (e.g. p=0, t=1)!"
                )

        if indexers:
            data, meta = self.isel(indexers, metadata=True)
            if Path(path).suffix not in {".tif", ".tiff"}:
                path = Path(path).with_suffix(".tiff")
            tifffile.imwrite(path, data, imagej=True)
            Path(path).with_suffix(".json").write_text(json.dumps(meta))
        else:
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            n_pos = len(self._tiff.series)
            for i in range(n_pos):
                data, meta = self.isel({"p": i}, metadata=True)
                tifffile.imwrite(out / f"p{i}.tif", data, imagej=True)
                (out / f"p{i}.json").write_text(json.dumps(meta))

    def close(self) -> None:
        """Close the TIFF file and release resources."""
        if hasattr(self, "_tiff") and self._tiff is not None:
            self._tiff.close()
            self._tiff = None  # type: ignore[assignment]
        self._metadata_list = []
        self._sequence = None  # type: ignore[assignment]

    # -------------------------Private Methods-------------------------

    def _build_sequence(self) -> useq.MDASequence:
        """Build MDASequence from OME metadata."""
        # Check for plate metadata
        if self._ome.plates:
            ome_plate = self._ome.plates[0]

            # Determine FOVs per well from first well's samples
            fovs = 1
            if ome_plate.wells:
                fovs = max(len(ome_plate.wells[0].well_samples), 1)

            self._plate_plan = ome_plate_to_plate_plan(ome_plate, fovs_per_well=fovs)
            stage_positions: useq.WellPlatePlan | list[useq.Position] = self._plate_plan
        else:
            # No plate - create positions from images
            positions = []
            for i, img in enumerate(self._ome.images):
                name = img.name or f"p{i:04d}"
                positions.append(useq.Position(name=name))
            stage_positions = positions

        return build_sequence(
            stage_positions=stage_positions,
            size_t=self._size_t,
            size_z=self._size_z,
            size_c=self._size_c,
            time_increment=self._time_increment,
            channel_names=self._channel_names,
            exposure=self._exposure_ms or 100.0,
        )

    def _build_metadata(self) -> list[dict]:
        """Build frame-level metadata from OME metadata."""
        meta_list = []
        n_positions = len(self._tiff.series)

        for p in range(n_positions):
            for t in range(self._size_t):
                for c in range(self._size_c):
                    for z in range(self._size_z):
                        # Get position name
                        if p < len(self._ome.images):
                            pos_name = self._ome.images[p].name or f"p{p:04d}"
                        elif self._sequence and p < len(self._sequence.stage_positions):
                            pos = self._sequence.stage_positions[p]
                            pos_name = pos.name or f"p{p:04d}"
                        else:
                            pos_name = f"p{p:04d}"

                        frame_meta: dict[str, Any] = {
                            EVENT_KEY: {
                                "index": {"p": p, "t": t, "c": c, "z": z},
                                "pos_name": pos_name,
                            },
                        }
                        if self._exposure_ms is not None:
                            frame_meta["exposure_ms"] = self._exposure_ms
                        if self._pixel_size_um is not None:
                            frame_meta["pixel_size_um"] = self._pixel_size_um

                        meta_list.append(frame_meta)

        return meta_list

    def _build_array_index(
        self, indexers: Mapping[str, int], axes: str
    ) -> tuple[int | slice, ...]:
        """Build numpy-style index tuple from indexers and axes string.

        Parameters
        ----------
        indexers : Mapping[str, int]
            Axis-index mapping (p, t, c, z).
        axes : str
            Axes string from tifffile series (e.g. "TCYX", "TZCYX").
        """
        # Map tifffile axis chars to our indexer keys
        axis_map = {"T": "t", "C": "c", "Z": "z"}
        result: list[int | slice] = []
        for ax in axes.upper():
            key = axis_map.get(ax)
            if key and key in indexers:
                result.append(indexers[key])
            elif ax in ("X", "Y"):
                result.append(slice(None))
            else:
                result.append(slice(None))
        return tuple(result)

    def _get_metadata_for_indexers(self, indexers: Mapping[str, int]) -> list[dict]:
        """Return metadata entries matching the given indexers."""
        result = []
        for meta in self._metadata_list:
            event_index = meta[EVENT_KEY]["index"]
            if indexers.items() <= event_index.items():
                result.append(meta)
        return result
