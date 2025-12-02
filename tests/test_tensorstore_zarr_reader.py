import json
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
import useq
import zarr

from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader


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
