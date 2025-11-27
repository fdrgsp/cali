"""Tests for TiffCollectionReader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from cali.readers import TiffCollectionReader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_tiff_files(tmp_path: Path) -> dict[str, list[Path]]:
    """Create temporary TIFF files for testing."""
    file_map = {}

    # Create files for well A1 (3 FOVs)
    a1_files = []
    for p in range(3):
        filename = f"A1_fov{p:04d}.tif"
        filepath = tmp_path / filename
        data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        tifffile.imwrite(filepath, data)
        a1_files.append(filepath)
    file_map["A1"] = a1_files

    # Create files for well A2 (2 FOVs)
    a2_files = []
    for p in range(2):
        filename = f"A2_fov{p:04d}.tif"
        filepath = tmp_path / filename
        data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        tifffile.imwrite(filepath, data)
        a2_files.append(filepath)
    file_map["A2"] = a2_files

    return file_map


def test_tiff_collection_reader_basic(temp_tiff_files: dict[str, list[Path]]) -> None:
    """Test basic TiffCollectionReader functionality."""
    reader = TiffCollectionReader(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    )

    # Check sequence - WellPlatePlan creates uniform grid (2 wells × 3 max FOVs = 6)
    assert len(reader.sequence.stage_positions) == 6

    # Check data access
    result = reader.isel({"p": 0, "t": 0})
    assert isinstance(result, np.ndarray)
    assert result.shape == (64, 64)

    # Check metadata - only 5 actual files (A1:3, A2:2)
    assert len(reader.metadata) == 5


def test_tiff_collection_reader_with_metadata(
    temp_tiff_files: dict[str, list[Path]],
) -> None:
    """Test TiffCollectionReader with metadata retrieval."""
    reader = TiffCollectionReader(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    )

    _, metadata = reader.isel({"p": 0, "t": 0}, metadata=True)
    assert len(metadata) > 0
    assert "exposure_ms" in metadata[0]
    assert metadata[0]["exposure_ms"] == 100.0
    assert "mda_event" in metadata[0]


def test_tiff_collection_reader_position_names(
    temp_tiff_files: dict[str, list[Path]],
) -> None:
    """Test that position names are correctly generated."""
    reader = TiffCollectionReader(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    )

    # Check position names (A1_0000, A1_0001, A1_0002, A2_0000, A2_0001)
    assert reader.sequence.stage_positions[0].name == "A1_0000"
    assert reader.sequence.stage_positions[1].name == "A1_0001"
    assert reader.sequence.stage_positions[2].name == "A1_0002"
    assert reader.sequence.stage_positions[3].name == "A2_0000"
    assert reader.sequence.stage_positions[4].name == "A2_0001"


def test_tiff_collection_reader_coverslip(tmp_path: Path) -> None:
    """Test TiffCollectionReader with coverslip plate."""
    # Create files for single well
    files = []
    for p in range(4):
        filename = f"fov{p:04d}.tif"
        filepath = tmp_path / filename
        data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        tifffile.imwrite(filepath, data)
        files.append(filepath)

    file_map = {"A1": files}

    reader = TiffCollectionReader(
        file_map=file_map,
        plate="coverslip-22mm-square",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    )

    assert len(reader.sequence.stage_positions) == 4
    assert reader.sequence.stage_positions[0].name == "A1_0000"


def test_tiff_collection_reader_write_tiff(
    temp_tiff_files: dict[str, list[Path]], tmp_path: Path
) -> None:
    """Test writing TIFF from reader."""
    reader = TiffCollectionReader(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    )

    output_path = tmp_path / "output.tif"
    reader.write_tiff(str(output_path), indexers={"p": 0, "t": 0})

    assert output_path.exists()
    # Verify we can read it back
    data = tifffile.imread(output_path)
    assert data.shape == (64, 64)
