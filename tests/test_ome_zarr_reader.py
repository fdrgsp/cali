import json
from pathlib import Path

import numpy as np
import pytest
import useq
import zarr

from cali.readers._ome_zarr_reader import OMEZarrReader


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
