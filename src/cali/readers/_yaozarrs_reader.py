"""Reader for OME-Zarr files using yaozarrs.

Uses yaozarrs for metadata parsing and lazy zarr array access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import useq
import yaozarrs

from cali._constants import EVENT_KEY
from cali.readers._ome_to_useq import (
    build_sequence,
    ngff_plate_to_plate_plan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


class YaozarrsReader:
    """Reader for OME-Zarr files using yaozarrs with lazy array access.

    Reads OME-Zarr files following the NGFF specification (v0.4+).
    Plate/well metadata is extracted from zarr attributes and converted
    to useq.WellPlatePlan for GUI compatibility.

    Parameters
    ----------
    path : str | Path
        Path to the .zarr directory.

    Examples
    --------
    >>> reader = YaozarrsReader("experiment.zarr")
    >>> data = reader.isel(p=0, t=5, c=0)
    >>> data, meta = reader.isel(p=0, t=5, metadata=True)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._group = yaozarrs.open_group(self._path)

        # Parse OME metadata
        self._ome_meta = self._group.ome_metadata()

        # Determine structure: plate-based or image-based
        self._plate_plan: useq.WellPlatePlan | None = None
        self._image_paths: list[str] = []  # paths to image arrays within zarr
        self._axis_names: list[str] = []  # e.g. ["t", "c", "z", "y", "x"]

        self._parse_structure()

        # Build sequence and metadata
        self._sequence = self._build_sequence()
        self._metadata_list = self._build_metadata()

    # -------------------------Properties-------------------------

    @property
    def path(self) -> Path:
        """Return the path to the zarr directory."""
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
        if p >= len(self._image_paths):
            raise IndexError(
                f"Position index {p} out of range "
                f"(have {len(self._image_paths)} images)."
            )

        # Navigate to the image array
        image_path = self._image_paths[p]
        image_group = self._group.get(image_path)
        if image_group is None:
            raise ValueError(f"Image path not found in zarr: {image_path}")

        # Get highest resolution array (first dataset)
        arr = self._get_array(image_group)

        # Build index
        idx = self._build_array_index(indexers)
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
        import tifffile

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
            for i in range(len(self._image_paths)):
                data, meta = self.isel({"p": i}, metadata=True)
                tifffile.imwrite(out / f"p{i}.tif", data, imagej=True)
                (out / f"p{i}.json").write_text(json.dumps(meta))

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "_group"):
            self._group = None  # type: ignore[assignment]
        self._metadata_list = []
        self._sequence = None  # type: ignore[assignment]

    # -------------------------Private Methods-------------------------

    def _parse_structure(self) -> None:
        """Parse the NGFF zarr structure to find images and plate layout."""
        attrs = self._group.attrs

        if "plate" in attrs:
            self._parse_plate_structure(attrs["plate"])
        elif "multiscales" in attrs:
            # Single image (no plate) - the root IS the image
            self._image_paths = [""]
            self._parse_axes_from_multiscales(attrs["multiscales"])
        else:
            raise ValueError(
                f"No plate or multiscales metadata found in {self._path}. "
                "This does not appear to be a valid NGFF OME-Zarr."
            )

    def _parse_plate_structure(self, plate_attrs: dict[str, Any]) -> None:
        """Parse plate-level NGFF structure."""
        wells = plate_attrs.get("wells", [])
        if not wells:
            raise ValueError("Plate metadata has no wells.")

        # Collect all image paths in well order
        fovs_per_well = 0
        for well_info in wells:
            well_path = well_info["path"]  # e.g. "A/1"
            well_group = self._group.get(well_path)
            if well_group is None:
                continue

            well_attrs = well_group.attrs
            if "well" not in well_attrs:
                continue

            images = well_attrs["well"].get("images", [])
            fovs_per_well = max(fovs_per_well, len(images))

            for img_info in images:
                img_path = img_info["path"]  # e.g. "0"
                full_path = f"{well_path}/{img_path}"
                self._image_paths.append(full_path)

                # Parse axes from first image's multiscales
                if not self._axis_names:
                    img_group = self._group.get(full_path)
                    if img_group is not None:
                        ms = img_group.attrs.get("multiscales")
                        if ms:
                            self._parse_axes_from_multiscales(ms)

        # Build plate plan
        self._plate_plan = ngff_plate_to_plate_plan(
            plate_attrs, fovs_per_well=max(fovs_per_well, 1)
        )

    def _parse_axes_from_multiscales(self, multiscales: list[dict[str, Any]]) -> None:
        """Extract axis names from multiscales metadata."""
        if multiscales and "axes" in multiscales[0]:
            self._axis_names = [
                ax.get("name", ax.get("type", "")) for ax in multiscales[0]["axes"]
            ]

    def _get_dimension_size(self, axis: str) -> int:
        """Get the size of a given axis from the first image array."""
        if not self._image_paths:
            return 1
        if axis not in self._axis_names:
            return 1

        image_group = self._group.get(self._image_paths[0])
        if image_group is None:
            return 1

        arr = self._get_array(image_group)
        axis_idx = self._axis_names.index(axis)
        return int(arr.shape[axis_idx]) if axis_idx < len(arr.shape) else 1

    def _get_array(self, image_group: yaozarrs.ZarrGroup) -> Any:
        """Get the highest-resolution array from an image group.

        NGFF images store multi-resolution arrays as "0", "1", etc.
        "0" is always the highest resolution.
        """
        arr_node = image_group.get("0")
        if arr_node is None:
            # Fallback: maybe it's already an array
            return image_group.to_zarr_python()
        return arr_node.to_zarr_python()

    def _build_sequence(self) -> useq.MDASequence:
        """Build MDASequence from parsed NGFF structure."""
        size_t = self._get_dimension_size("t")
        size_z = self._get_dimension_size("z")
        size_c = self._get_dimension_size("c")

        if self._plate_plan is not None:
            stage_positions: useq.WellPlatePlan | list[useq.Position] = self._plate_plan
        else:
            positions = [
                useq.Position(name=f"p{i:04d}") for i in range(len(self._image_paths))
            ]
            stage_positions = positions

        return build_sequence(
            stage_positions=stage_positions,
            size_t=size_t,
            size_z=size_z,
            size_c=size_c,
        )

    def _build_metadata(self) -> list[dict]:
        """Build frame-level metadata."""
        meta_list = []
        size_t = self._get_dimension_size("t")
        size_c = self._get_dimension_size("c")
        size_z = self._get_dimension_size("z")

        for p in range(len(self._image_paths)):
            # Get position name
            if self._sequence and p < len(self._sequence.stage_positions):
                pos = self._sequence.stage_positions[p]
                pos_name = pos.name or f"p{p:04d}"
            else:
                pos_name = f"p{p:04d}"

            for t in range(size_t):
                for c in range(size_c):
                    for z in range(size_z):
                        frame_meta: dict[str, Any] = {
                            EVENT_KEY: {
                                "index": {"p": p, "t": t, "c": c, "z": z},
                                "pos_name": pos_name,
                            },
                        }
                        meta_list.append(frame_meta)

        return meta_list

    def _build_array_index(
        self, indexers: Mapping[str, int]
    ) -> tuple[int | slice, ...]:
        """Build index tuple from indexers and axis names."""
        result: list[int | slice] = []
        for ax in self._axis_names:
            if ax in indexers:
                result.append(indexers[ax])
            elif ax in ("x", "y"):
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
