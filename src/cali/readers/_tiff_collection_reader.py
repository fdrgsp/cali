"""Virtual zarr-like reader for collections of TIFF files.

This module provides a reader that maps TIFF files to a plate/well/FOV structure
and provides lazy array-like access without loading everything into memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tifffile
import useq

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


@dataclass
class TiffCollectionSettings:
    """Settings for TiffCollectionReader.

    This dataclass encapsulates all configuration needed to create a
    TiffCollectionReader. It can be used to easily pass configuration
    between functions or store it for later reconstruction.

    Attributes
    ----------
    file_map : dict[str, list[Path | str]]
        Mapping from well names to lists of TIFF file paths
    plate : useq.WellPlate | str
        Plate definition
    metadata : dict[str, Any]
        Metadata with 'exposure_ms' and 'pixel_size_um'
    tiff_folder_path : str | Path
        The path to the folder(s) containing the TIFF files.
    """

    file_map: Mapping[str, list[Path | str]]
    plate: useq.WellPlate | str
    metadata: dict[str, Any]
    tiff_folder_path: str | Path


class TiffCollectionReader:
    """Virtual zarr-like reader for collections of TIFF files.

    Maps TIFF files to wells/FOVs and provides lazy array-like access
    without loading everything into memory. Compatible with the cali pipeline.

    Can be called in two ways:
    Pass the settings as a `TiffCollectionSettings` instance or as individual
    parameters dictionary.

    Parameters
    ----------
    settings : TiffCollectionSettings | dict[str, list[Path | str]] | None
        Either a TiffCollectionSettings instance, a file_map dict, or None.
        If TiffCollectionSettings, other parameters are ignored.
        If dict, this is treated as file_map and other params must be provided.

    Example:
    >>> settings = TiffCollectionSettings(
    ...     file_map=file_map,
    ...     plate="96-well",
    ...     metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    ...      "tiff_folder_path": "/path/to/tiff_folder",
    ... )
    >>> reader = TiffCollectionReader(settings)
    Or:
    >>> reader = TiffCollectionReader(
    ...     {
    ...      "file_map": file_map,
    ...      "plate": "96-well",
    ...      "metadata": {"exposure_ms": 100.0, "pixel_size_um": 0.65},
    ...      "tiff_folder_path": "/path/to/tiff_folder",
    ... }
    )
    """

    def __init__(
        self,
        settings: TiffCollectionSettings | dict[str, Any] | None = None,
    ) -> None:
        try:
            if isinstance(settings, dict):
                settings = TiffCollectionSettings(**settings)

            if isinstance(settings, TiffCollectionSettings):
                file_map = settings.file_map
                plate = settings.plate
                metadata = settings.metadata
                tiff_folder_path = settings.tiff_folder_path
            else:
                raise TypeError(
                    f"First argument must be TiffCollectionSettings, dict, or None, "
                    f"got {type(settings)}"
                )
        except TypeError as e:
            raise TypeError(
                "If passing a dict as the first argument, it must contain "
                "'file_map', 'plate', 'metadata', and 'data_path' keys!"
            ) from e

        if not file_map:
            raise ValueError("file_map cannot be empty")

        # Verify all wells have the same number of FOVs
        fov_counts = {well: len(files) for well, files in file_map.items()}
        unique_counts = set(fov_counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"All wells must have the same number of FOVs. "
                f"Found varying counts: {fov_counts}"
            )

        # Verify all files exist
        missing_files = self._check_for_missing_files(file_map, tiff_folder_path)
        if missing_files:
            raise FileNotFoundError(
                f"TIFF files not found in tiff_folder_path or as absolute paths: "
                f"{missing_files}"
            )

        # Store original file_map for later export
        self._original_file_map = file_map

        # Inspect first TIFF to determine structure (time, channel, z dimensions)
        first_file = Path(next(iter(next(iter(file_map.values())))))
        with tifffile.TiffFile(first_file) as tif:
            shape = tif.asarray().shape
            # Assume shape is (T, Y, X), (T, C, Y, X), or (T, Z, Y, X), etc.
            # For now, handle (T, Y, X) - multi-frame time series
            if len(shape) == 3:
                # (T, Y, X) - time series
                num_frames = shape[0]
                self._max_t = num_frames
                self._max_c = 1
                self._max_z = 1
            elif len(shape) == 2:
                # (Y, X) - single frame
                self._max_t = 1
                self._max_c = 1
                self._max_z = 1
            else:
                # More complex - would need metadata to determine axis order
                raise NotImplementedError(
                    f"TIFF shape {shape} not yet supported. "
                    "Only (T, Y, X) and (Y, X) formats are currently supported."
                )

        # Convert to Path objects and build position mapping
        self._file_mapping: dict[tuple[int, ...], Path] = {}
        self._well_to_position: dict[str, list[int]] = {}

        position_idx = 0
        for well_name, tiff_files in file_map.items():
            position_indices = []
            for tiff_file in tiff_files:
                # Each TIFF file represents one position
                # Store mapping from (p, t, c, z) to file path
                # File contains all timepoints, map each t to same file
                tiff_path = Path(tiff_file)
                for t in range(self._max_t):
                    self._file_mapping[(position_idx, t, 0, 0)] = tiff_path
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

        # Construct plate plan and sequence
        self._plate_plan, self._time_plan, self._z_plan, self._channels = (
            self._build_plate_plan()
        )
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

    def close(self) -> None:
        """Close the reader and release any resources.

        For TiffCollectionReader, this mainly clears cached metadata
        to free memory. Individual TIFF files are opened/closed on demand.
        """
        if hasattr(self, "_metadata"):
            self._metadata = []
        if hasattr(self, "_sequence"):
            self._sequence = None
        if hasattr(self, "_plate_plan"):
            self._plate_plan = None

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

        # Find the TIFF file and frame index
        # If 't' not specified, frame_idx will be None (load entire time series)
        result = self._find_tiff_for_index(indexers)
        if result is None:
            raise ValueError(f"No TIFF file found for indexers: {indexers}")

        tiff_path, frame_idx = result
        # Lazy load the data (full file if frame_idx is None, specific frame otherwise)
        data = self._load_tiff(tiff_path, frame_idx)

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
        file_map: Mapping[str, list[str | Path]],
        tiff_folder_path: str | Path,
    ) -> list[str]:
        """Check for missing files in the collection."""
        missing_files = []
        for _, files in file_map.items():
            for file_path in files:
                # Check if file exists in any the provided tiff_folder_path
                if not (Path(tiff_folder_path) / file_path).exists():
                    missing_files.append(str(file_path))
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
                    "pos_name": pos_name,
                },
                "file_path": str(tiff_path),
            }
            meta_list.append(frame_meta)

        return meta_list

    def _build_plate_plan(
        self,
    ) -> tuple[useq.WellPlatePlan, Any, Any, Any]:
        """Build a useq.WellPlatePlan from the file collection.

        Returns
        -------
        tuple[useq.WellPlatePlan, Any, Any, Any]
            (plate_plan, time_plan, z_plan, channels)
        """
        # Build time/z/channel plans
        time_plan = (
            useq.TIntervalLoops(interval=timedelta(seconds=0), loops=self._max_t)
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
        # Determine which wells are used
        well_indices = []
        for well_name in self._well_to_position.keys():
            row = ord(well_name[0]) - ord("A")  # Convert A->0, B->1, etc.
            col = int(well_name[1:]) - 1  # Convert 1->0, 2->1, etc.
            well_indices.append((row, col))
        # Find FOVs per well (all wells have same number of FOVs due to validation)
        max_fovs = len(next(iter(self._original_file_map.values())))
        well_points_plan = useq.RandomPoints(num_points=max_fovs)
        plan_kwargs: dict[str, Any] = {
            "plate": self._plate,
            "a1_center_xy": (0, 0),
            "selected_wells": (
                tuple(w[0] for w in well_indices),
                tuple(w[1] for w in well_indices),
            ),
            "well_points_plan": well_points_plan,
        }
        if time_plan:
            plan_kwargs["time_plan"] = time_plan
        if z_plan:
            plan_kwargs["z_plan"] = z_plan
        if channels:
            plan_kwargs["channels"] = channels

        return useq.WellPlatePlan(**plan_kwargs), time_plan, z_plan, channels

    def _build_sequence_from_plan(self) -> useq.MDASequence:
        """Build MDASequence from the plate plan with properly named positions."""
        # Generate the sequence from the plate plan
        # Include time_plan, z_plan, and channels
        sequence_kwargs: dict[str, Any] = {
            "stage_positions": self._plate_plan,
        }
        if self._time_plan:
            sequence_kwargs["time_plan"] = self._time_plan
        if self._z_plan:
            sequence_kwargs["z_plan"] = self._z_plan
        if self._channels:
            sequence_kwargs["channels"] = self._channels

        return useq.MDASequence(**sequence_kwargs)

    def _find_tiff_for_index(
        self, indexers: Mapping[str, int]
    ) -> tuple[Path, int | None] | None:
        """Find the TIFF file and frame index for the given indexers.

        Returns
        -------
        tuple[Path, int | None] | None
            (file_path, frame_index) or None if not found.
            For multi-frame TIFFs, frame_index is the timepoint within the file.
            If 't' is not in indexers, frame_index is None (load entire time series).
        """
        # Build index tuple
        p = indexers.get("p", 0)
        c = indexers.get("c", 0)
        z = indexers.get("z", 0)

        # If 't' not specified, return None for frame_idx to load entire time series
        if "t" not in indexers:
            # Use t=0 just to look up the file path (all t values map to same file)
            tiff_path = self._file_mapping.get((p, 0, c, z))
            if tiff_path is None:
                return None
            return tiff_path, None

        # Otherwise, return specific frame index
        t = indexers["t"]
        tiff_path = self._file_mapping.get((p, t, c, z))
        if tiff_path is None:
            return None
        return tiff_path, t

    def _load_tiff(self, tiff_path: Path, frame_idx: int | None = 0) -> np.ndarray:
        """Load a specific frame from a TIFF file using memory mapping.

        Uses tifffile.memmap to create a memory-mapped array that only loads
        data from disk when accessed. For multi-frame TIFFs, extracts the
        requested frame (timepoint).

        Parameters
        ----------
        tiff_path : Path
            Path to the TIFF file.
        frame_idx : int | None
            Frame index to load (for time series). If None, load entire file.
            Default is 0.

        Returns
        -------
        np.ndarray
            Memory-mapped array for the requested frame or entire file.
        """
        data = tifffile.memmap(tiff_path, mode="r")
        # If frame_idx is None, return entire file
        if frame_idx is None:
            return data
        # If multi-frame (3D: T, Y, X), extract the requested frame
        if len(data.shape) == 3:
            return data[frame_idx]
        # If single frame (2D: Y, X), return as-is
        return data

    def _get_metadata_from_index(self, indexers: Mapping[str, int]) -> list[dict]:
        """Return the metadata for the given indexers."""
        result = []
        for meta in self._metadata:
            event_index = meta["mda_event"]["index"]
            if indexers.items() <= event_index.items():
                result.append(meta)
        return result
