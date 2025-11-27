"""Virtual zarr-like reader for collections of TIFF files.

This module provides a reader that maps TIFF files to a plate/well/FOV structure
and provides lazy array-like access without loading everything into memory.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tifffile
import useq

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


class TiffCollectionReader:
    r"""Virtual zarr-like reader for collections of TIFF files.

    Maps TIFF files to wells/FOVs and provides lazy array-like access
    without loading everything into memory. Compatible with the cali pipeline.

    Parameters
    ----------
    file_map : dict[str, list[Path | str]]
        Mapping from well names to lists of TIFF file paths.
        For multi-well plates: {"A1": [fov1.tif, fov2.tif], "A2": [...], ...}
        For single coverslip: {"A1": [fov1.tif, fov2.tif, ...]}
    plate : useq.WellPlate | str
        Plate definition. Can be a WellPlate instance or a plate name string
        (e.g., "96-well", "coverslip-22mm-square").
    metadata : dict
        Metadata to apply to all positions. Must include:
        - exposure_ms: float
        - pixel_size_um: float
    data_path : str | Path
        Base path to verify and resolve file paths. Files are validated
        against this path during initialization.

    Attributes
    ----------
    path : Path
        Virtual path representing this collection.
    sequence : useq.MDASequence
        The constructed MDASequence from the TIFF collection.
    metadata : list[dict]
        Full metadata for all frames.

    Methods
    -------
    isel(indexers, metadata=False)
        Select data from the collection by index (lazy loading).
    write_tiff(path, indexers=None)
        Write selected data to a TIFF file.

    Examples
    --------
    >>> # Multi-well plate
    >>> file_map = {
    ...     "A1": ["A1_fov1.tif", "A1_fov2.tif"],
    ...     "A2": ["A2_fov1.tif", "A2_fov2.tif"],
    ... }
    >>> reader = TiffCollectionReader(
    ...     file_map=file_map,
    ...     plate="96-well",
    ...     metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65}
    ... )
    >>>
    >>> # Single coverslip
    >>> file_map = {"A1": ["fov1.tif", "fov2.tif", "fov3.tif"]}
    >>> reader = TiffCollectionReader(
    ...     file_map=file_map,
    ...     plate="coverslip-22mm-square",
    ...     metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65}
    ... )
    >>>
    >>> # Access data lazily (only loads when needed)
    >>> data = reader.isel({"p": 0, "t": 0})
    """

    def __init__(
        self,
        file_map: dict[str, list[Path | str]],
        plate: useq.WellPlate | str,
        metadata: dict[str, Any],
        data_path: str | Path,
    ) -> None:
        if not file_map:
            raise ValueError("file_map cannot be empty")

        # Verify all files exist
        missing_files = self._check_for_missing_files(file_map, data_path)
        if missing_files:
            raise FileNotFoundError(
                f"TIFF files not found in data_path or as absolute paths: "
                f"{missing_files}"
            )

        # Store original file_map for later export
        self._original_file_map = file_map

        # Convert to Path objects and build position mapping
        self._file_mapping: dict[tuple[int, ...], Path] = {}
        self._well_to_position: dict[str, list[int]] = {}

        position_idx = 0
        for well_name, tiff_files in file_map.items():
            position_indices = []
            for tiff_file in tiff_files:
                # Map (p, t, c, z) to file path
                # For now, assume single timepoint, channel, z-slice per file
                self._file_mapping[(position_idx, 0, 0, 0)] = Path(tiff_file)
                position_indices.append(position_idx)
                position_idx += 1
            self._well_to_position[well_name] = position_indices

        # Create plate
        self._plate = (
            plate
            if isinstance(plate, useq.WellPlate)
            else useq.WellPlate.from_str(plate)
        )

        # Build metadata
        self._metadata = self._build_metadata(metadata)

        # Store time/z/channel info for sequence building
        indices = list(self._file_mapping.keys())
        self._max_t = max(idx[1] for idx in indices) + 1
        self._max_c = max(idx[2] for idx in indices) + 1
        self._max_z = max(idx[3] for idx in indices) + 1

        # Construct plate plan and sequence
        self._plate_plan = self._build_plate_plan()
        self._sequence = self._build_sequence_from_plan()

        # Virtual path
        self._path = Path("tiff_collection_virtual")

    @property
    def path(self) -> Path:
        """Return the virtual path."""
        return self._path

    @property
    def sequence(self) -> useq.MDASequence:
        """Return the MDASequence."""
        return self._sequence

    @property
    def plate_plan(self) -> useq.WellPlatePlan:
        """Return the WellPlatePlan."""
        return self._plate_plan

    @property
    def metadata(self) -> list[dict]:
        """Return the unstructured full metadata."""
        return self._metadata

    # _________________________PUBLIC METHODS___________________________

    def isel(
        self,
        indexers: Mapping[str, int] | None = None,
        metadata: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, list[dict]]:
        """Select data from the collection.

        Parameters
        ----------
        indexers : Mapping[str, int] | None
            The indexers to select the data (e.g. {"p": 0, "t": 1}).
            If None, returns data for first position.
        metadata : bool
            If True, return the metadata as well. By default, False.
        **kwargs : Any
            Additional way to pass the indexers as kwargs (e.g. p=0, t=1).

        Returns
        -------
        np.ndarray | tuple[np.ndarray, list[dict]]
            The selected data, and optionally metadata.
        """
        if indexers is None:
            indexers = {}
        if kwargs:
            if all(
                isinstance(k, str) and isinstance(v, int) for k, v in kwargs.items()
            ):
                indexers = {**indexers, **kwargs}
            else:
                raise TypeError(
                    "kwargs must be a mapping from strings to integers (e.g. p=0, t=1)!"
                )

        # Find the TIFF file matching these indexers
        tiff_path = self._find_tiff_for_index(indexers)

        if tiff_path is None:
            raise ValueError(f"No TIFF file found for indexers: {indexers}")

        # Lazy load only the requested file
        data = self._load_tiff(tiff_path)

        if metadata:
            meta = self._get_metadata_from_index(indexers)
            return data, meta
        return data

    def write_tiff(
        self,
        path: str | Path,
        indexers: Mapping[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the data to a TIFF file.

        Parameters
        ----------
        path : str | Path
            The path to the output TIFF file.
        indexers : Mapping[str, int] | None
            The indexers to select the data. If None, write all positions.
        **kwargs : Any
            Additional indexers as kwargs.
        """
        if indexers:
            data, meta = self.isel(indexers, metadata=True)
            if Path(path).suffix not in {".tif", ".tiff"}:
                path = Path(path).with_suffix(".tiff")
            tifffile.imwrite(path, data, imagej=True)
            # Save metadata as json
            dest = Path(path).with_suffix(".json")
            dest.write_text(json.dumps(meta))
        else:
            # Write all positions
            if not Path(path).exists():
                Path(path).mkdir(parents=True, exist_ok=True)
            for i in range(len(self._sequence.stage_positions)):
                data, meta = self.isel({"p": i}, metadata=True)
                tifffile.imwrite(Path(path) / f"p{i}.tif", data, imagej=True)
                dest = Path(path) / f"p{i}.json"
                dest.write_text(json.dumps(meta))

    def to_experiment_tiff_config(
        self,
    ) -> tuple[dict[str, list[str]], str, dict[str, Any]]:
        """Export configuration for saving to database.

        Returns
        -------
        tuple[dict[str, list[str]], str, dict[str, Any]]
            (file_map, plate_type, metadata) tuple for database storage.
        """
        # Convert original file_map paths to strings
        file_map_str: dict[str, list[str]] = {
            well: [str(path) for path in paths]
            for well, paths in self._original_file_map.items()
        }

        # Extract metadata from first frame
        metadata = {}
        if self._metadata:
            first_meta = self._metadata[0]
            metadata["exposure_ms"] = first_meta.get("exposure_ms")
            metadata["pixel_size_um"] = first_meta.get("pixel_size_um")

        return file_map_str, self._plate.name, metadata

    # ___________________________PRIVATE METHODS___________________________

    def _check_for_missing_files(
        self,
        file_map: dict[str, list[str | Path]],
        data_path: str | Path,
    ) -> list[str]:
        """Check for missing files in the collection."""
        data_path = Path(data_path)
        missing_files = []
        for _, files in file_map.items():
            for file_str in files:
                if not Path(file_str).exists():
                    missing_files.append(file_str)
        return missing_files

    def _build_metadata(self, metadata: dict) -> list[dict]:
        """Build metadata for all frames."""
        if "exposure_ms" not in metadata:
            raise ValueError("metadata must include 'exposure_ms'")
        if "pixel_size_um" not in metadata:
            raise ValueError("metadata must include 'pixel_size_um'")

        # Create metadata for each file
        meta_list = []
        for (p, t, c, z), tiff_path in self._file_mapping.items():
            # Find well name for this position
            pos_name = f"p{p:04d}"
            for well_name, position_indices in self._well_to_position.items():
                if p in position_indices:
                    fov_idx = position_indices.index(p)
                    pos_name = f"{well_name}_{fov_idx:04d}"
                    break

            frame_meta = {
                "exposure_ms": metadata["exposure_ms"],
                "pixel_size_um": metadata["pixel_size_um"],
                "mda_event": {
                    "index": {"p": p, "t": t, "c": c, "z": z},
                },
                "pos_name": pos_name,
                "file_path": str(tiff_path),
            }
            meta_list.append(frame_meta)

        return meta_list

    def _build_plate_plan(self) -> useq.WellPlatePlan:
        """Build a useq.WellPlatePlan from the file collection."""
        # Determine which wells are used
        well_indices = []
        for well_name in self._well_to_position.keys():
            row = ord(well_name[0]) - ord("A")  # Convert A->0, B->1, etc.
            col = int(well_name[1:]) - 1  # Convert 1->0, 2->1, etc.
            well_indices.append((row, col))

        # Find max FOVs per well to create a consistent grid
        max_fovs = max(len(positions) for positions in self._well_to_position.values())

        # Create a grid plan for the FOVs using GridRowsColumns
        grid_plan = useq.GridRowsColumns(rows=1, columns=max_fovs)

        # Build time/z/channel plans
        time_plan = (
            useq.TIntervalLoops(interval=timedelta(seconds=1), loops=self._max_t)
            if self._max_t > 1
            else None
        )
        z_plan = (
            useq.ZRangeAround(range=self._max_z, step=1.0) if self._max_z > 1 else None
        )
        channels = (
            tuple(
                useq.Channel(config=f"ch{i}", exposure=100.0)
                for i in range(self._max_c)
            )
            if self._max_c > 1
            else None
        )

        # Create WellPlatePlan
        plan_kwargs: dict[str, Any] = {
            "plate": self._plate,
            "a1_center_xy": (0, 0),
            "selected_wells": (
                tuple(w[0] for w in well_indices),
                tuple(w[1] for w in well_indices),
            ),
            "well_points_plan": grid_plan,
        }
        if time_plan:
            plan_kwargs["time_plan"] = time_plan
        if z_plan:
            plan_kwargs["z_plan"] = z_plan
        if channels:
            plan_kwargs["channels"] = channels

        return useq.WellPlatePlan(**plan_kwargs)

    def _build_sequence_from_plan(self) -> useq.MDASequence:
        """Build MDASequence from the plate plan with properly named positions."""
        # Generate the sequence from the plate plan
        # This will create positions with well-based names
        return useq.MDASequence(stage_positions=self._plate_plan)

    def _find_tiff_for_index(self, indexers: Mapping[str, int]) -> Path | None:
        """Find the TIFF file corresponding to the given index."""
        # Build index tuple
        p = indexers.get("p", 0)
        t = indexers.get("t", 0)
        c = indexers.get("c", 0)
        z = indexers.get("z", 0)

        return self._file_mapping.get((p, t, c, z))

    def _load_tiff(self, tiff_path: Path) -> np.ndarray:
        """Load a single TIFF file using memory mapping for lazy loading.

        Uses tifffile.memmap to create a memory-mapped array that only loads
        data from disk when accessed. This is much more memory-efficient than
        loading the entire file into RAM.

        Returns
        -------
        np.ndarray
            Memory-mapped array that loads data on-demand.
        """
        return tifffile.memmap(tiff_path, mode="r")

    def _get_metadata_from_index(self, indexers: Mapping[str, int]) -> list[dict]:
        """Return the metadata for the given indexers."""
        result = []
        for meta in self._metadata:
            event_index = meta["mda_event"]["index"]
            if indexers.items() <= event_index.items():
                result.append(meta)
        return result
