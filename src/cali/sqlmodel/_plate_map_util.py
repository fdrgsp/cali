"""Utility functions for plate map hashing and comparison."""

from __future__ import annotations

import hashlib
import json


def compute_plate_map_hash(plate_maps: dict[str, dict[str, str]] | None) -> str | None:
    """Compute a stable hash from plate_maps dictionary.

    This hash is used to detect when plate map configurations change between runs.
    Two runs with identical technical extraction settings but different plate maps
    should create separate CaliResult entries.

    Parameters
    ----------
    plate_maps : dict[str, dict[str, str]] | None
        Plate map configuration mapping condition names to well positions.
        Format: {"genotype": {"A1": "WT", "A2": "KO", ...},
                 "treatment": {"A1": "Vehicle", "A2": "Drug", ...}}

    Returns
    -------
    str | None
        SHA256 hash of the sorted plate_maps dict, or None if plate_maps is None.

    Examples
    --------
    >>> plate_maps = {
    ...     "genotype": {"A1": "WT", "A2": "KO"},
    ...     "treatment": {"A1": "Vehicle", "A2": "Drug"},
    ... }
    >>> hash1 = compute_plate_map_hash(plate_maps)
    >>> hash2 = compute_plate_map_hash(plate_maps)
    >>> hash1 == hash2  # Same input produces same hash
    True

    >>> different_maps = {
    ...     "genotype": {"A1": "KO", "A2": "WT"}  # Swapped values
    ... }
    >>> hash3 = compute_plate_map_hash(different_maps)
    >>> hash1 == hash3  # Different input produces different hash
    False
    """
    if plate_maps is None:
        return None

    # Sort keys at all levels for stable hashing
    sorted_maps = {
        condition: dict(sorted(wells.items()))
        for condition, wells in sorted(plate_maps.items())
    }

    # Convert to JSON string with sorted keys
    json_str = json.dumps(sorted_maps, sort_keys=True)

    # Compute SHA256 hash
    return hashlib.sha256(json_str.encode()).hexdigest()
