from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from useq import MDASequence, WellPlatePlan

from cali.readers._tiff_collection_reader import (
    TiffCollectionReader,
    TiffCollectionSettings,
)


@pytest.fixture
def temp_tiff_files(tmp_path: Path) -> dict[str, list[Path]]:
    """Create temporary TIFF files for testing."""
    # Create a simple 2-position, 2-timepoint experiment
    # Well A1: 2 FOVs (p0, p1)
    # Each FOV has 2 timepoints (t0, t1)
    # Files: A1_fov0001.tif (contains t0, t1), A1_fov0002.tif (contains t0, t1)

    # Create dummy TIFF data (T=2, Y=10, X=10)
    data = np.zeros((2, 10, 10), dtype=np.uint16)

    files = {}
    well_dir = tmp_path

    # Create files for A1
    p0_path = well_dir / "A1_fov0001.tif"
    tifffile.imwrite(p0_path, data)

    p1_path = well_dir / "A1_fov0002.tif"
    tifffile.imwrite(p1_path, data)

    files["A1"] = [p0_path, p1_path]

    return files


def test_init_invalid_type() -> None:
    """Test initialization with invalid type."""
    # The error message is re-raised with a different message in __init__
    with pytest.raises(TypeError, match="If passing a dict as the first argument"):
        TiffCollectionReader("invalid")


def test_init_empty_file_map(tmp_path: Path) -> None:
    """Test initialization with empty file map."""
    settings = TiffCollectionSettings(
        file_map={},
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    with pytest.raises(ValueError, match="file_map cannot be empty"):
        TiffCollectionReader(settings)


def test_init_missing_files(tmp_path: Path) -> None:
    """Test initialization with missing files."""
    settings = TiffCollectionSettings(
        file_map={"A1": ["missing.tif"]},
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="TIFF files not found"):
        TiffCollectionReader(settings)


def test_init_unsupported_shape(tmp_path: Path) -> None:
    """Test initialization with unsupported TIFF shape."""
    # Create 4D TIFF (C, T, Y, X) which might be unsupported
    # Based on code, it seems to expect (T, Y, X) or (Y, X)

    p0_path = tmp_path / "A1_fov0001.tif"
    # Create 4D data
    data = np.zeros((2, 2, 10, 10), dtype=np.uint16)
    tifffile.imwrite(p0_path, data)

    settings = TiffCollectionSettings(
        file_map={"A1": [p0_path]},
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )

    # This might raise ValueError during validation if shape is checked
    # Or it might pass if validation is loose. Let's see.
    # Reading the code, _validate_tiff_files checks shape.
    # It expects len(shape) in (2, 3).

    with pytest.raises(NotImplementedError, match="not yet supported"):
        TiffCollectionReader(settings)


def test_to_experiment_tiff_config(
    temp_tiff_files: dict[str, list[Path]], tmp_path: Path
) -> None:
    """Test export of configuration."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    file_map, plate_name, metadata = reader.to_experiment_tiff_config()

    assert "A1" in file_map
    assert len(file_map["A1"]) == 2
    assert plate_name == "96-well"
    assert metadata["exposure_ms"] == 100.0
    assert metadata["pixel_size_um"] == 0.5


def test_properties(temp_tiff_files: dict[str, list[Path]], tmp_path: Path) -> None:
    """Test properties."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    assert isinstance(reader.path, Path)
    assert isinstance(reader.sequence, MDASequence)
    assert isinstance(reader.plate_plan, WellPlatePlan)
    assert isinstance(reader.metadata, list)
    assert len(reader.metadata) > 0


def test_isel_metadata(temp_tiff_files: dict[str, list[Path]], tmp_path: Path) -> None:
    """Test isel with metadata=True."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    # Select p=0, t=0
    data, meta = reader.isel({"p": 0, "t": 0}, metadata=True)

    assert isinstance(data, np.ndarray)
    assert isinstance(meta, list)
    assert len(meta) > 0
    assert meta[0]["exposure_ms"] == 100.0


def test_isel_invalid_kwargs(
    temp_tiff_files: dict[str, list[Path]], tmp_path: Path
) -> None:
    """Test isel with invalid kwargs."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    with pytest.raises(TypeError, match="kwargs must be a mapping"):
        reader.isel(invalid="value")


def test_isel_missing_file(
    temp_tiff_files: dict[str, list[Path]], tmp_path: Path
) -> None:
    """Test isel with indexers that don't match any file."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    # p=99 does not exist
    with pytest.raises(ValueError, match="No TIFF file found"):
        reader.isel({"p": 99})


def test_write_tiff(temp_tiff_files: dict[str, list[Path]], tmp_path: Path) -> None:
    """Test write_tiff."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    output_path = tmp_path / "output.tif"

    # Mock tifffile.imwrite to avoid actual writing and verify calls
    with patch("tifffile.imwrite") as mock_imwrite:
        # Test writing specific index
        reader.write_tiff(output_path, {"p": 0, "t": 0})
        mock_imwrite.assert_called_once()

        # Test writing all positions
        mock_imwrite.reset_mock()
        output_dir = tmp_path / "output_dir"
        reader.write_tiff(output_dir)
        assert mock_imwrite.call_count == 2  # 2 positions


def test_missing_metadata_fields(
    temp_tiff_files: dict[str, list[Path]], tmp_path: Path
) -> None:
    """Test missing metadata fields."""
    # Missing exposure_ms
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    with pytest.raises(ValueError, match="metadata must include 'exposure_ms'"):
        TiffCollectionReader(settings)

    # Missing pixel_size_um
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0},
        tiff_folder_path=tmp_path,
    )
    with pytest.raises(ValueError, match="metadata must include 'pixel_size_um'"):
        TiffCollectionReader(settings)


def test_load_tiff_full_file(
    temp_tiff_files: dict[str, list[Path]], tmp_path: Path
) -> None:
    """Test loading full TIFF file (frame_idx=None)."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    # Use internal method to test frame_idx=None
    # We need to find a valid path first
    tiff_path = temp_tiff_files["A1"][0]
    data = reader._load_tiff(tiff_path, frame_idx=None)

    # Should return the full stack (T=2, Y=10, X=10)
    assert data.shape == (2, 10, 10)


def test_find_tiff_no_t(temp_tiff_files: dict[str, list[Path]], tmp_path: Path) -> None:
    """Test _find_tiff_for_index without 't' indexer."""
    settings = TiffCollectionSettings(
        file_map=temp_tiff_files,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.5},
        tiff_folder_path=tmp_path,
    )
    reader = TiffCollectionReader(settings)

    # Should return path and None for frame_idx
    result = reader._find_tiff_for_index({"p": 0})
    assert result is not None
    path, frame_idx = result
    assert path == temp_tiff_files["A1"][0]
    assert frame_idx is None
