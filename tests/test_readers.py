import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorstore as ts
import tifffile
import useq
import zarr
from useq import MDASequence, WellPlatePlan

from cali.readers._ome_zarr_reader import OMEZarrReader
from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader
from cali.readers._tiff_collection_reader import (
    TiffCollectionReader,
    TiffCollectionSettings,
)


@pytest.fixture
def mock_tensorstore_zarr(tmp_path: Path) -> Path:
    """Create a mock Tensorstore Zarr structure."""
    path = tmp_path / "test_ts.zarr"

    # Create sequence
    seq = useq.MDASequence(
        stage_positions=[(0, 0, 0), (10, 10, 0)],
        time_plan={"interval": 1, "loops": 5},
        channels=["GFP"],
        axis_order="tpcz",  # Explicit axis order without xy
    )

    # Create data array (TPCZYX)
    # 5 timepoints, 2 positions, 1 channel, 1 z, 100 y, 100 x
    shape = (5, 2, 1, 1, 100, 100)
    data = np.zeros(shape, dtype=np.uint16)

    # Fill data
    for t in range(5):
        for p in range(2):
            data[t, p, 0, 0, :, :] = t + p * 10

    # Create array at root
    store = zarr.open_array(str(path), mode="w", shape=shape, dtype=np.uint16)
    store[:] = data

    # Add metadata
    frame_metas = []
    for t in range(5):
        for p in range(2):
            frame_metas.append({"mda_event": {"index": {"t": t, "p": p}}})

    store.attrs["frame_metadatas"] = frame_metas
    store.attrs["useq_MDASequence"] = seq.model_dump_json()

    return path


def test_tensorstore_zarr_reader_init(mock_tensorstore_zarr: Path) -> None:
    """Test initialization."""
    reader = TensorstoreZarrReader(mock_tensorstore_zarr)
    assert reader.path == mock_tensorstore_zarr
    assert isinstance(reader.store, ts.TensorStore)
    assert isinstance(reader.sequence, useq.MDASequence)


def test_tensorstore_zarr_reader_metadata(mock_tensorstore_zarr: Path) -> None:
    """Test metadata retrieval."""
    reader = TensorstoreZarrReader(mock_tensorstore_zarr)
    meta = reader.metadata

    # The reader returns the full metadata dict if useq_MDASequence is present
    if isinstance(meta, dict):
        assert "frame_metadatas" in meta
        assert len(meta["frame_metadatas"]) == 10
    else:
        assert len(meta) == 10


def test_tensorstore_zarr_reader_isel(mock_tensorstore_zarr: Path) -> None:
    """Test isel method."""
    reader = TensorstoreZarrReader(mock_tensorstore_zarr)

    # Select p=0, t=2
    # Expected value: t + p*10 = 2 + 0 = 2
    data = reader.isel({"p": 0, "t": 2})
    assert isinstance(data, np.ndarray)
    assert data.shape == (100, 100)
    assert np.all(data == 2)

    # Select p=1, t=4
    # Expected value: 4 + 10 = 14
    data = reader.isel(p=1, t=4)
    assert np.all(data == 14)

    # Test with metadata
    data, meta = reader.isel(p=0, t=0, metadata=True)
    assert isinstance(meta, list)
    # Should contain metadata for this frame
    # The reader implementation filters metadata based on indexers


def test_tensorstore_zarr_reader_write_tiff(
    mock_tensorstore_zarr: Path, tmp_path: Path
) -> None:
    """Test write_tiff method."""
    reader = TensorstoreZarrReader(mock_tensorstore_zarr)
    output_path = tmp_path / "output.tif"

    # Write specific frame
    reader.write_tiff(output_path, p=0, t=0)
    assert output_path.exists()

    # Write all frames (not supported or different behavior?)
    # The implementation seems similar to OMEZarrReader.
    output_dir = tmp_path / "output_dir"
    reader.write_tiff(output_dir)
    assert output_dir.exists()
    # It iterates over positions if sequence is present?
    # Or it writes the whole array?
    # The code says:
    # if indexers: ...
    # else: ...
    # if self.sequence: ... iterate positions ...

    # Since we have sequence, it should write p0.tif, p1.tif etc.
    assert (output_dir / "p0.tif").exists()
    assert (output_dir / "p1.tif").exists()


def test_tensorstore_zarr_reader_errors(mock_tensorstore_zarr: Path) -> None:
    """Test error handling in TensorstoreZarrReader."""
    reader = TensorstoreZarrReader(mock_tensorstore_zarr)

    # Test invalid kwargs in isel
    with pytest.raises(TypeError, match="kwargs must be a mapping"):
        reader.isel(invalid_arg=1.5)  # type: ignore

    # Test invalid axis in indexers
    with pytest.raises(ValueError, match="Invalid axis"):
        reader.isel(k=0)  # 'k' is not in axis_order for the mock data

    # Test invalid kwargs in write_tiff
    with pytest.raises(TypeError, match="kwargs must be a mapping"):
        reader.write_tiff("out.tif", invalid_arg=1.5)  # type: ignore


def test_tensorstore_zarr_reader_init_with_store(mock_tensorstore_zarr: Path) -> None:
    """Test initialization with a TensorStore object."""
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(mock_tensorstore_zarr)},
    }
    store = ts.open(spec).result()
    reader = TensorstoreZarrReader(store)
    assert reader.path == mock_tensorstore_zarr
    assert reader.store is not None


def test_tensorstore_zarr_reader_no_sequence(tmp_path: Path) -> None:
    """Test reader with no sequence metadata."""
    path = tmp_path / "no_seq.zarr"

    # Create empty zarr array
    zarr.open_array(str(path), mode="w", shape=(10, 10), dtype=np.uint16)

    # Open with tensorstore to write metadata
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
    }
    store = ts.open(spec).result()

    # Write empty metadata
    store.kvstore.write(".zattrs", json.dumps({}).encode()).result()

    reader = TensorstoreZarrReader(path)
    assert reader.sequence is None

    # Test write_tiff raises error without sequence
    with pytest.raises(ValueError, match=r"No 'useq.MDASequence' found"):
        reader.write_tiff(tmp_path / "out")


# ============================================================================
# OME-Zarr Reader Tests
# ============================================================================


@pytest.fixture
def mock_ome_zarr(tmp_path: Path) -> Path:
    """Create a mock OME-Zarr structure."""
    path = tmp_path / "test.zarr"
    store = zarr.open(str(path), mode="w")

    # Create sequence
    seq = useq.MDASequence(
        stage_positions=[(0, 0, 0), (10, 10, 0)],
        time_plan={"interval": 1, "loops": 5},
        channels=["GFP"],
    )

    # Create data array (TCZYX)
    # 5 timepoints, 1 channel, 1 z, 100 y, 100 x
    data = np.zeros((5, 1, 1, 100, 100), dtype=np.uint16)
    # Fill with some data to verify reading
    for t in range(5):
        data[t, 0, 0, :, :] = t

    # Create position groups (as datasets directly for this reader?)
    # If the reader expects store[pos_key] to be sliceable, it must be an array.
    p0 = store.create_dataset("p0", data=data)
    p0.attrs["useq_MDASequence"] = json.loads(seq.model_dump_json())
    p0.attrs["frame_meta"] = [
        {"mda_event": {"index": {"t": i, "p": 0}}} for i in range(5)
    ]
    p0.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

    p1 = store.create_dataset("p1", data=data)
    p1.attrs["useq_MDASequence"] = json.loads(seq.model_dump_json())
    p1.attrs["frame_meta"] = [
        {"mda_event": {"index": {"t": i, "p": 1}}} for i in range(5)
    ]
    p1.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

    return path


def test_ome_zarr_reader_init(mock_ome_zarr: Path) -> None:
    """Test initialization."""
    reader = OMEZarrReader(mock_ome_zarr)
    assert reader.path == mock_ome_zarr
    assert isinstance(reader.store, zarr.Group)
    assert isinstance(reader.sequence, useq.MDASequence)


def test_ome_zarr_reader_metadata(mock_ome_zarr: Path) -> None:
    """Test metadata retrieval."""
    reader = OMEZarrReader(mock_ome_zarr)
    meta = reader.metadata()
    assert len(meta) == 10  # 2 positions * 5 frames
    # meta is a list of dicts like {"mda_event": {"index": {"t": i, "p": 0}}}
    # The order depends on iteration order of keys, which might be p0 then p1 or vice
    # versa. But let's check content.

    # Check that we have entries for p=0 and p=1
    p0_count = sum(1 for m in meta if m["mda_event"]["index"]["p"] == 0)
    p1_count = sum(1 for m in meta if m["mda_event"]["index"]["p"] == 1)
    assert p0_count == 5
    assert p1_count == 5


def test_ome_zarr_reader_isel(mock_ome_zarr: Path) -> None:
    """Test isel method."""
    reader = OMEZarrReader(mock_ome_zarr)

    # Select specific frame
    # Data was created with shape (5, 1, 1, 100, 100) -> TCZYX
    # And filled with t value.
    # isel should handle mapping from axis names to indices.
    # Assuming standard OME-Zarr axes or useq sequence axes.
    # The reader uses _get_axis_index which likely maps t, c, z to indices.

    # Select p=0, t=2
    data = reader.isel({"p": 0, "t": 2})
    assert isinstance(data, np.ndarray)
    # Should be 2D (YX) or 3D if C/Z are kept?
    # The code says .squeeze(), so likely 2D if C=1, Z=1.
    assert data.shape == (100, 100)
    assert np.all(data == 2)

    # Select p=1, t=4
    data = reader.isel(p=1, t=4)
    assert np.all(data == 4)

    # Test with metadata
    data, meta = reader.isel(p=0, t=0, metadata=True)
    assert isinstance(meta, list)
    # meta should contain frame_meta for that frame
    # In mock, frame_meta is a list of dicts.
    # _get_metadata_from_index likely filters it.

    # Test error when p is missing and multiple positions exist
    with pytest.raises(ValueError, match="should contain the 'p' axis"):
        reader.isel(t=0)


def test_ome_zarr_reader_write_tiff(mock_ome_zarr: Path, tmp_path: Path) -> None:
    """Test write_tiff method."""
    reader = OMEZarrReader(mock_ome_zarr)
    output_path = tmp_path / "output.tif"

    # Write specific frame
    reader.write_tiff(output_path, p=0, t=0)
    assert output_path.exists()

    # Write whole position stack
    output_stack = tmp_path / "stack.tif"
    reader.write_tiff(output_stack, p=0)
    assert output_stack.exists()

    # Write all positions (directory mode)
    output_dir = tmp_path / "output_dir"
    reader.write_tiff(output_dir)
    assert output_dir.exists()
    assert output_dir.is_dir()

    # Should contain p0.tif, p1.tif etc.
    assert (output_dir / "p0.tif").exists()
    assert (output_dir / "p1.tif").exists()
    assert output_dir.is_dir()
    # Should contain tiff files for each frame?
    # The implementation iterates over keys.
    # If indexers are None, it writes all data per position?
    # Let's check implementation.
    # if indexers: ... imwrite ...
    # else: ... mkdir ... for key in keys ... imwrite ...

    # Check if files exist in output_dir
    # It seems it writes to output_dir / f"{pos_key}.tif" ?
    # Or maybe it iterates over timepoints?
    # The code says:
    # if pos := len(keys): ...
    # for key in keys: ... imwrite(path / f"{key}.tif", ...)
    # keys are p0, p1.
    # So it writes p0.tif, p1.tif inside output_dir.

    assert (output_dir / "p0.tif").exists()
    assert (output_dir / "p1.tif").exists()


# ============================================================================
# TIFF Collection Reader Tests
# ============================================================================


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
