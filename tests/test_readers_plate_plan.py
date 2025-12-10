"""Tests for plate_plan functionality in readers."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
import useq
import zarr

from cali.readers._ome_zarr_reader import OMEZarrReader
from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader


@pytest.fixture
def mock_tensorstore_zarr_with_sequence(tmp_path: Path) -> Path:
    """Create a mock Tensorstore Zarr structure with sequence metadata."""
    path = tmp_path / "test_ts.zarr"

    # Create sequence with simple stage positions
    seq = useq.MDASequence(
        stage_positions=[(0, 0, 0), (10, 10, 0)],  # type: ignore[arg-type]
        time_plan={"interval": 1, "loops": 3},  # type: ignore[arg-type]
        channels=["GFP"],  # type: ignore[arg-type]
        axis_order="tpcz",  # type: ignore[arg-type]
    )

    # Create data array (TPCZYX)
    shape = (3, 2, 1, 1, 50, 50)
    data = np.zeros(shape, dtype=np.uint16)

    for t in range(3):
        for p in range(2):
            data[t, p, 0, 0, :, :] = t + p * 10

    # Create array at root
    store = zarr.open_array(str(path), mode="w", shape=shape, dtype=np.uint16)
    store[:] = data

    # Add metadata - use the new format with mda_event containing sequence
    frame_metas = []
    for t in range(3):
        for p in range(2):
            frame_metas.append(
                {
                    "mda_event": {
                        "index": {"t": t, "p": p},
                        "sequence": seq.model_dump(),
                    }
                }
            )

    store.attrs["frame_metadatas"] = frame_metas

    return path


@pytest.fixture
def mock_ome_zarr_with_sequence(tmp_path: Path) -> Path:
    """Create a mock OME-Zarr structure with sequence metadata."""
    path = tmp_path / "test.zarr"
    store = zarr.open(str(path), mode="w")  # type: ignore[assignment]

    # Create sequence
    seq = useq.MDASequence(
        stage_positions=[(0, 0, 0), (10, 10, 0)],  # type: ignore[arg-type]
        time_plan={"interval": 1, "loops": 3},  # type: ignore[arg-type]
        channels=["GFP"],  # type: ignore[arg-type]
    )

    # Create data array (TCZYX)
    data = np.zeros((3, 1, 1, 50, 50), dtype=np.uint16)
    for t in range(3):
        data[t, 0, 0, :, :] = t

    # Create position groups
    p0 = store.create_dataset("p0", data=data)  # type: ignore[attr-defined]
    p0.attrs["useq_MDASequence"] = json.loads(seq.model_dump_json())
    p0.attrs["frame_meta"] = [
        {"mda_event": {"index": {"t": i, "p": 0}}} for i in range(3)
    ]
    p0.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

    p1 = store.create_dataset("p1", data=data)  # type: ignore[attr-defined]
    p1.attrs["useq_MDASequence"] = json.loads(seq.model_dump_json())
    p1.attrs["frame_meta"] = [
        {"mda_event": {"index": {"t": i, "p": 1}}} for i in range(3)
    ]
    p1.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

    return path


def test_tensorstore_zarr_reader_with_plate_plan(
    mock_tensorstore_zarr_with_sequence: Path,
) -> None:
    """Test TensorstoreZarrReader with plate_plan parameter."""
    # Create a well plate plan
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0, 1), (0, 1)),  # A1, A2, B1, B2
        well_points_plan=useq.RandomPoints(num_points=2),
    )

    # Create reader with plate_plan
    reader = TensorstoreZarrReader(
        mock_tensorstore_zarr_with_sequence, plate_plan=plate_plan
    )

    assert reader.sequence is not None
    # Plate plan generates: 2 wells (A1, B2) x 2 points = 4 positions
    assert len(reader.sequence.stage_positions) == 4

    # Verify the sequence was replaced, not the original 2 positions
    original_reader = TensorstoreZarrReader(mock_tensorstore_zarr_with_sequence)
    assert len(original_reader.sequence.stage_positions) == 2  # type: ignore


def test_ome_zarr_reader_with_plate_plan(mock_ome_zarr_with_sequence: Path) -> None:
    """Test OMEZarrReader with plate_plan parameter."""
    # Create a well plate plan
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0, 1), (0, 1)),  # A1, A2, B1, B2
        well_points_plan=useq.RandomPoints(num_points=2),
    )

    # Create reader with plate_plan
    reader = OMEZarrReader(mock_ome_zarr_with_sequence, plate_plan=plate_plan)

    assert reader.sequence is not None
    # Plate plan generates: 2 wells (A1, B2) x 2 points = 4 positions
    assert len(reader.sequence.stage_positions) == 4

    # Verify the sequence was replaced, not the original 2 positions
    original_reader = OMEZarrReader(mock_ome_zarr_with_sequence)
    assert len(original_reader.sequence.stage_positions) == 2  # type: ignore


def test_tensorstore_zarr_reader_set_plate_plan(
    mock_tensorstore_zarr_with_sequence: Path,
) -> None:
    """Test set_plate_plan method for TensorstoreZarrReader."""
    reader = TensorstoreZarrReader(mock_tensorstore_zarr_with_sequence)
    assert reader.sequence is not None
    original_positions = len(reader.sequence.stage_positions)
    assert original_positions == 2

    # Create and set a plate plan
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("24-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0,), (0, 1)),  # A1, A2
        well_points_plan=useq.RandomPoints(num_points=3),
    )

    reader.set_plate_plan(plate_plan)

    assert reader.sequence is not None
    # 1 row x 2 cols x 3 points = 6 positions
    assert len(reader.sequence.stage_positions) == 6


def test_ome_zarr_reader_set_plate_plan(mock_ome_zarr_with_sequence: Path) -> None:
    """Test set_plate_plan method for OMEZarrReader."""
    reader = OMEZarrReader(mock_ome_zarr_with_sequence)
    assert reader.sequence is not None
    original_positions = len(reader.sequence.stage_positions)
    assert original_positions == 2

    # Create and set a plate plan
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("24-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0,), (0, 1)),  # A1, A2
        well_points_plan=useq.RandomPoints(num_points=3),
    )

    reader.set_plate_plan(plate_plan)

    assert reader.sequence is not None
    # 1 row x 2 cols x 3 points = 6 positions
    assert len(reader.sequence.stage_positions) == 6


def test_tensorstore_zarr_reader_set_plate_plan_no_sequence(tmp_path: Path) -> None:
    """Test set_plate_plan raises error when sequence is None."""
    path = tmp_path / "no_seq.zarr"

    # Create zarr without sequence metadata
    zarr.open_array(str(path), mode="w", shape=(10, 10), dtype=np.uint16)
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
    }
    store = ts.open(spec).result()
    store.kvstore.write(  # type: ignore[union-attr]
        ".zattrs", json.dumps({"frame_metadatas": []}).encode()
    ).result()

    reader = TensorstoreZarrReader(path)
    assert reader.sequence is None

    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0,), (0,)),
    )

    with pytest.raises(ValueError, match=r"No 'useq.MDASequence' found"):
        reader.set_plate_plan(plate_plan)


def test_ome_zarr_reader_set_plate_plan_no_sequence(tmp_path: Path) -> None:
    """Test set_plate_plan raises error when sequence is None."""
    path = tmp_path / "no_seq.zarr"

    # Create zarr without sequence metadata
    store = zarr.open(str(path), mode="w")  # type: ignore[assignment]
    # Create a position without useq_MDASequence
    p0 = store.create_dataset(  # type: ignore[attr-defined]
        "p0", data=np.zeros((5, 1, 1, 50, 50), dtype=np.uint16)
    )
    p0.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

    reader = OMEZarrReader(path)
    assert reader.sequence is None

    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0,), (0,)),
    )

    with pytest.raises(ValueError, match=r"No 'useq.MDASequence' found"):
        reader.set_plate_plan(plate_plan)


def test_tensorstore_zarr_reader_with_no_hcs_data() -> None:
    """Test TensorstoreZarrReader with actual no_hcs test data."""
    path = Path(__file__).parent / "test_data" / "no_hcs" / "no_hcs.tensorstore.zarr"

    if not path.exists():
        pytest.skip("no_hcs test data not found")

    # Test reading without plate plan
    reader = TensorstoreZarrReader(path)
    assert reader.sequence is not None
    original_positions = len(reader.sequence.stage_positions)
    assert original_positions > 0

    # Test with plate plan
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((6,), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),  # G2 to G11
        well_points_plan=useq.RandomPoints(num_points=2),
    )

    reader_with_plate = TensorstoreZarrReader(path, plate_plan=plate_plan)
    assert reader_with_plate.sequence is not None
    # 1 row x 10 cols x 2 points = 20 positions
    assert len(reader_with_plate.sequence.stage_positions) == 20

    # Verify original reader is unchanged
    assert len(reader.sequence.stage_positions) == original_positions


def test_tensorstore_zarr_reader_plate_plan_updates_positions_correctly(
    mock_tensorstore_zarr_with_sequence: Path,
) -> None:
    """Test that plate_plan actually generates correct well positions."""
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(100.0, 200.0),
        selected_wells=((0,), (0, 1)),  # A1, A2
        well_points_plan=useq.RandomPoints(num_points=1),
    )

    reader = TensorstoreZarrReader(
        mock_tensorstore_zarr_with_sequence, plate_plan=plate_plan
    )

    assert reader.sequence is not None
    positions = reader.sequence.stage_positions
    # 1 row x 2 cols x 1 point = 2 positions
    assert len(positions) == 2

    # Check that positions are Position objects with x, y coordinates
    for pos in positions:
        assert hasattr(pos, "x")
        assert hasattr(pos, "y")


def test_ome_zarr_reader_plate_plan_updates_positions_correctly(
    mock_ome_zarr_with_sequence: Path,
) -> None:
    """Test plate_plan generates correct well positions for OMEZarrReader."""
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(100.0, 200.0),
        selected_wells=((0,), (0, 1)),  # A1, A2
        well_points_plan=useq.RandomPoints(num_points=1),
    )

    reader = OMEZarrReader(mock_ome_zarr_with_sequence, plate_plan=plate_plan)

    assert reader.sequence is not None
    positions = reader.sequence.stage_positions
    # 1 row x 2 cols x 1 point = 2 positions
    assert len(positions) == 2

    # Check that positions are Position objects with x, y coordinates
    for pos in positions:
        assert hasattr(pos, "x")
        assert hasattr(pos, "y")
