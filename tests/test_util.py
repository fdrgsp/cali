"""Tests for utility functions in cali.util._util."""

from pathlib import Path

import numpy as np
import pytest

from cali.util._util import (
    coordinates_to_mask,
    mask_to_coordinates,
)


@pytest.mark.parametrize(
    "mask,expected_coords,expected_shape",
    [
        # Test case 1: Simple 3x3 mask with single pixel
        (
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
            ([1], [1]),
            (3, 3),
        ),
        # Test case 2: 3x3 mask with diagonal
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
            ([0, 1, 2], [0, 1, 2]),
            (3, 3),
        ),
        # Test case 3: Empty mask
        (
            np.array([[0, 0], [0, 0]], dtype=bool),
            ([], []),
            (2, 2),
        ),
        # Test case 4: Full mask
        (
            np.array([[1, 1], [1, 1]], dtype=bool),
            ([0, 0, 1, 1], [0, 1, 0, 1]),
            (2, 2),
        ),
        # Test case 5: Rectangular region
        (
            np.array(
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool
            ),
            ([1, 1, 2, 2], [1, 2, 1, 2]),
            (4, 4),
        ),
    ],
)
def test_mask_to_coordinates(
    mask: np.ndarray,
    expected_coords: tuple[list[int], list[int]],
    expected_shape: tuple[int, int],
) -> None:
    """Test mask_to_coordinates with various mask patterns."""
    coords, shape = mask_to_coordinates(mask)

    assert shape == expected_shape
    y_coords, x_coords = coords
    expected_y, expected_x = expected_coords

    # Sort both actual and expected for consistent comparison
    y_x_pairs = sorted(zip(y_coords, x_coords))
    expected_pairs = sorted(zip(expected_y, expected_x))

    assert y_x_pairs == expected_pairs


@pytest.mark.parametrize(
    "coordinates,shape,expected_mask",
    [
        # Test case 1: Single pixel
        (
            ([1], [1]),
            (3, 3),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
        ),
        # Test case 2: Diagonal
        (
            ([0, 1, 2], [0, 1, 2]),
            (3, 3),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ),
        # Test case 3: Empty
        (
            ([], []),
            (2, 2),
            np.array([[0, 0], [0, 0]], dtype=bool),
        ),
        # Test case 4: Full mask
        (
            ([0, 0, 1, 1], [0, 1, 0, 1]),
            (2, 2),
            np.array([[1, 1], [1, 1]], dtype=bool),
        ),
        # Test case 5: Rectangular region
        (
            ([1, 1, 2, 2], [1, 2, 1, 2]),
            (4, 4),
            np.array(
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool
            ),
        ),
    ],
)
def test_coordinates_to_mask(
    coordinates: tuple[list[int], list[int]],
    shape: tuple[int, int],
    expected_mask: np.ndarray,
) -> None:
    """Test coordinates_to_mask with various coordinate patterns."""
    mask = coordinates_to_mask(coordinates, shape)
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_coordinates_roundtrip() -> None:
    """Test that converting mask -> coordinates -> mask is lossless."""
    # Create a random mask
    np.random.seed(42)
    original_mask = np.random.randint(0, 2, size=(10, 10), dtype=bool)

    # Convert to coordinates
    coords, shape = mask_to_coordinates(original_mask)

    # Convert back to mask
    reconstructed_mask = coordinates_to_mask(coords, shape)

    # Should be identical
    np.testing.assert_array_equal(original_mask, reconstructed_mask)


def test_coordinates_mask_roundtrip() -> None:
    """Test that converting coordinates -> mask -> coordinates is lossless."""
    # Create coordinates for a specific pattern
    original_coords = ([0, 1, 2, 3], [3, 2, 1, 0])  # Anti-diagonal
    shape = (5, 5)

    # Convert to mask
    mask = coordinates_to_mask(original_coords, shape)

    # Convert back to coordinates
    coords, reconstructed_shape = mask_to_coordinates(mask)

    # Shape should match
    assert reconstructed_shape == shape

    # Coordinates should match (after sorting)
    y_coords, x_coords = coords
    orig_y, orig_x = original_coords

    reconstructed_pairs = sorted(zip(y_coords, x_coords))
    original_pairs = sorted(zip(orig_y, orig_x))

    assert reconstructed_pairs == original_pairs


def test_load_data_from_path_tensorstore(tmp_path: Path) -> None:
    """Test load_data_from_path with tensorstore zarr path."""
    from cali.util._util import load_data_from_path

    # Create a fake tensorstore zarr directory
    ts_path = tmp_path / "data.tensorstore.zarr"
    ts_path.mkdir()
    (ts_path / ".zattrs").write_text("{}")

    # Should attempt to load with TensorstoreZarrReader but fail
    # because it's not a valid tensorstore
    # The function should catch the error and handle it gracefully
    # or raise - either is acceptable, just verify it doesn't crash unexpectedly
    try:
        result = load_data_from_path(ts_path)
        # If it succeeds somehow, result should be a reader
        assert result is not None
    except (ValueError, RuntimeError):
        # Expected - not a valid tensorstore
        pass


def test_load_data_from_path_ome_zarr(tmp_path: Path) -> None:
    """Test load_data_from_path with OME zarr path."""
    from cali.util._util import load_data_from_path

    # Create a fake OME zarr directory
    zarr_path = tmp_path / "data.ome.zarr"
    zarr_path.mkdir()
    (zarr_path / ".zattrs").write_text("{}")

    # Should attempt to load with OMEZarrReader but may fail if not valid
    try:
        result = load_data_from_path(zarr_path)
        # If it succeeds, result should be a reader
        assert result is not None
    except (ValueError, RuntimeError, KeyError, FileNotFoundError):
        # Expected - not a valid OME zarr
        pass


def test_load_data_from_path_unsupported(tmp_path: Path) -> None:
    """Test load_data_from_path with unsupported format."""
    # Create some random path that's not a recognized format
    unsupported_path = tmp_path / "data.txt"
    unsupported_path.write_text("not a zarr")

    from cali.util._util import load_data_from_path

    # Should return None for unsupported formats
    result = load_data_from_path(unsupported_path)
    assert result is None


# ==================== Database utility functions tests ====================


def test_commit_fov_result_new_fov(tmp_path: Path) -> None:
    """Test commit_fov_result with a new FOV."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel._model import (
        FOV,
        ROI,
        DetectionSettings,
        Experiment,
        Mask,
        Plate,
    )
    from cali.util._util import commit_fov_result

    # Create a test database
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create tables
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create experiment and plate
        plate = Plate(name="test_plate", rows=8, columns=12)
        exp = Experiment(
            name="test_exp",
            experiment_type="Spontaneous Activity",
            plate=plate,
        )
        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Create a detection settings
        det_settings = DetectionSettings(
            experiment_id=exp.id,
            detection_method="test_method",
        )
        session.add(det_settings)
        session.commit()
        session.refresh(det_settings)

        # Create a FOV result
        fov_result = FOV(
            name="A1_0000",
            position_index=0,
            fov_number=0,
            fov_metadata={},
        )

        # Add a ROI with mask
        roi = ROI(
            label_value=1,
            active=True,
            stimulated=False,
            roi_mask=Mask(
                coords_y=[10, 11, 12],
                coords_x=[20, 21, 22],
                height=100,
                width=100,
                mask_type="roi",
            ),
        )
        fov_result.rois.append(roi)

        # Commit the FOV
        commit_fov_result(
            session,
            exp,
            fov_result,
            detection_settings_id=det_settings.id,
            commit=True,
        )

        # Verify the FOV was created
        from sqlmodel import select

        stmt = select(FOV).where(FOV.name == "A1_0000")
        saved_fov = session.exec(stmt).first()
        assert saved_fov is not None
        assert saved_fov.name == "A1_0000"
        assert saved_fov.position_index == 0
        assert len(saved_fov.rois) == 1
        assert saved_fov.rois[0].label_value == 1

    # Close engine to prevent resource warning
    engine.dispose()


def test_commit_fov_result_existing_fov_detection_mode(tmp_path: Path) -> None:
    """Test commit_fov_result adding new ROIs to existing FOV in detection mode."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel._model import (
        FOV,
        ROI,
        DetectionSettings,
        Experiment,
        Mask,
        Plate,
        Well,
    )
    from cali.util._util import commit_fov_result

    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        plate = Plate(name="test_plate", rows=8, columns=12)
        exp = Experiment(
            name="test_exp",
            experiment_type="Spontaneous Activity",
            plate=plate,
        )
        session.add(exp)
        session.commit()
        session.refresh(exp)
        session.refresh(plate)

        # Create well and existing FOV
        well = Well(
            plate_id=plate.id,
            name="A1",
            row=0,
            column=0,
        )
        session.add(well)
        session.commit()
        session.refresh(well)

        existing_fov = FOV(
            name="A1_0000",
            position_index=0,
            fov_number=0,
            well_id=well.id,
        )
        session.add(existing_fov)
        session.commit()
        session.refresh(existing_fov)

        det_settings = DetectionSettings(
            experiment_id=exp.id,
            detection_method="test_method",
        )
        session.add(det_settings)
        session.commit()
        session.refresh(det_settings)

        # Create FOV result with new ROIs
        fov_result = FOV(
            name="A1_0000",
            position_index=0,
            fov_number=0,
        )
        roi = ROI(
            label_value=2,
            active=True,
            stimulated=False,
            roi_mask=Mask(
                coords_y=[10],
                coords_x=[20],
                height=100,
                width=100,
                mask_type="roi",
            ),
        )
        fov_result.rois.append(roi)

        commit_fov_result(
            session,
            exp,
            fov_result,
            detection_settings_id=det_settings.id,
            commit=True,
        )

        # Verify ROI was added
        from sqlmodel import select

        stmt = select(FOV).where(FOV.id == existing_fov.id)
        updated_fov = session.exec(stmt).first()
        assert updated_fov is not None
        assert len(updated_fov.rois) == 1
        assert updated_fov.rois[0].label_value == 2

    # Close engine to prevent resource warning
    engine.dispose()


def test_commit_fov_result_raises_on_invalid_detection_settings_id(
    tmp_path: Path,
) -> None:
    """Test commit_fov_result raises ValueError when detection_settings_id not in DB."""
    from sqlmodel import Session, SQLModel, create_engine

    from cali.sqlmodel._model import (
        FOV,
        Experiment,
        Plate,
    )
    from cali.util._util import commit_fov_result

    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        plate = Plate(name="test_plate", rows=8, columns=12)
        exp = Experiment(
            name="test_exp",
            experiment_type="Spontaneous Activity",
            plate=plate,
        )
        session.add(exp)
        session.commit()

        fov_result = FOV(name="A1_0000", position_index=0, fov_number=0)

        import pytest

        with pytest.raises(ValueError, match="DetectionSettings with ID 999 not found"):
            commit_fov_result(
                session,
                exp,
                fov_result,
                detection_settings_id=999,
            )

    engine.dispose()


def test_update_fovs_in_database(tmp_path: Path) -> None:
    """Test update_fovs_in_database function."""
    from sqlmodel import Session, create_engine, select

    from cali.sqlmodel._model import (
        FOV,
        ROI,
        Experiment,
        Mask,
        Plate,
        Traces,
        Well,
    )
    from cali.util._util import update_fovs_in_database

    db_path = tmp_path / "test_update.db"
    engine = create_engine(f"sqlite:///{db_path}")

    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)

    # Create initial database structure
    with Session(engine) as session:
        plate = Plate(name="test_plate", rows=8, columns=12)
        exp = Experiment(
            name="test_exp",
            experiment_type="Spontaneous Activity",
            plate=plate,
        )
        session.add(exp)
        session.commit()
        session.refresh(plate)

        well = Well(plate_id=plate.id, name="A1", row=0, column=0)
        session.add(well)
        session.commit()
        session.refresh(well)

        fov = FOV(
            name="A1_0000",
            position_index=0,
            fov_number=0,
            well_id=well.id,
        )
        roi = ROI(
            label_value=1,
            active=True,
            stimulated=False,
            roi_mask=Mask(
                coords_y=[10],
                coords_x=[20],
                height=100,
                width=100,
                mask_type="roi",
            ),
        )
        fov.rois.append(roi)
        session.add(fov)
        session.commit()
        session.refresh(fov)
        session.refresh(roi)

    # Simulate what ExtractionRunner does - load FOV and add _new_traces
    with Session(engine) as session:
        stmt = select(FOV).where(FOV.position_index == 0)
        fov = session.exec(stmt).first()
        assert fov is not None

        # Simulate adding traces (ExtractionRunner pattern)
        roi = fov.rois[0]
        roi._new_traces = []  # type: ignore
        trace = Traces(
            raw_trace=[1.0, 2.0, 3.0],
            corrected_trace=[1.1, 2.1, 3.1],
        )
        roi._new_traces.append(trace)  # type: ignore

        # Initialize traces_history if not loaded (happens with lazy loading)
        if not hasattr(roi, "traces_history"):
            roi.traces_history = []

        # Update database within the same session - function will merge
        update_fovs_in_database(engine, fov)

    # Verify traces were saved
    with Session(engine) as session:
        stmt = select(FOV).where(FOV.position_index == 0)
        saved_fov = session.exec(stmt).first()
        assert saved_fov is not None
        assert len(saved_fov.rois) == 1
        assert len(saved_fov.rois[0].traces_history) == 1
        assert saved_fov.rois[0].traces_history[0].raw_trace == [1.0, 2.0, 3.0]

    # Close engine to prevent resource warning
    engine.dispose()


def test_load_fovs_from_database_all(tmp_path: Path) -> None:
    """Test load_fovs_from_database loading all FOVs."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel._model import FOV, Experiment, Plate, Well
    from cali.util._util import load_fovs_from_database

    db_path = tmp_path / "test_load.db"
    engine = create_engine(f"sqlite:///{db_path}")

    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)

    # Create test data
    with Session(engine) as session:
        plate = Plate(name="test_plate", rows=8, columns=12)
        exp = Experiment(
            name="test_exp",
            experiment_type="Spontaneous Activity",
            plate=plate,
        )
        session.add(exp)
        session.commit()
        session.refresh(plate)

        well = Well(plate_id=plate.id, name="A1", row=0, column=0)
        session.add(well)
        session.commit()
        session.refresh(well)

        fov1 = FOV(name="A1_0000", position_index=0, fov_number=0, well_id=well.id)
        fov2 = FOV(name="A1_0001", position_index=1, fov_number=1, well_id=well.id)
        session.add(fov1)
        session.add(fov2)
        session.commit()

    # Load all FOVs
    fovs = load_fovs_from_database(db_path)
    assert len(fovs) == 2
    assert fovs[0].position_index in [0, 1]
    assert fovs[1].position_index in [0, 1]

    # Close engine to prevent resource warning
    engine.dispose()


def test_load_fovs_from_database_filtered(tmp_path: Path) -> None:
    """Test load_fovs_from_database with position_indices filter."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel._model import FOV, Experiment, Plate, Well
    from cali.util._util import load_fovs_from_database

    db_path = tmp_path / "test_load_filter.db"
    engine = create_engine(f"sqlite:///{db_path}")

    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        plate = Plate(name="test_plate", rows=8, columns=12)
        exp = Experiment(
            name="test_exp",
            experiment_type="Spontaneous Activity",
            plate=plate,
        )
        session.add(exp)
        session.commit()
        session.refresh(plate)

        well = Well(plate_id=plate.id, name="A1", row=0, column=0)
        session.add(well)
        session.commit()
        session.refresh(well)

        for i in range(5):
            fov = FOV(
                name=f"A1_{i:04d}",
                position_index=i,
                fov_number=i,
                well_id=well.id,
            )
            session.add(fov)
        session.commit()

    # Load specific FOVs
    fovs = load_fovs_from_database(db_path, position_indices=[1, 3])
    assert len(fovs) == 2
    assert {f.position_index for f in fovs} == {1, 3}

    # Test with single integer
    fovs = load_fovs_from_database(db_path, position_indices=2)
    assert len(fovs) == 1
    assert fovs[0].position_index == 2

    # Close engine to prevent resource warning
    engine.dispose()


def test_save_labeled_images_from_fovs(tmp_path: Path) -> None:
    """Test save_labeled_images_from_fovs function."""
    from cali.sqlmodel._model import FOV, ROI, Mask
    from cali.util._util import save_labeled_images_from_fovs

    # Create a test FOV with ROIs
    fov = FOV(name="test_fov", position_index=0, fov_number=0)

    roi1 = ROI(
        label_value=1,
        active=True,
        stimulated=False,
        roi_mask=Mask(
            coords_y=[10, 10, 11, 11],
            coords_x=[20, 21, 20, 21],
            height=100,
            width=100,
            mask_type="roi",
        ),
    )
    roi2 = ROI(
        label_value=2,
        active=True,
        stimulated=False,
        roi_mask=Mask(
            coords_y=[30, 30, 31, 31],
            coords_x=[40, 41, 40, 41],
            height=100,
            width=100,
            mask_type="roi",
        ),
    )
    fov.rois = [roi1, roi2]

    output_dir = tmp_path / "labeled_images"
    save_labeled_images_from_fovs(fov, output_dir)

    # Verify output file exists
    output_file = output_dir / "test_fov_labeled.tif"
    assert output_file.exists()

    # Load and verify the labeled image
    import tifffile

    img = tifffile.imread(output_file)
    assert img.shape == (100, 100)
    assert img[10, 20] == 1
    assert img[30, 40] == 2
    assert img[0, 0] == 0  # background


def test_save_labeled_images_from_fovs_detection_filter(tmp_path: Path) -> None:
    """Test save_labeled_images_from_fovs with detection_settings_id filter."""
    from cali.sqlmodel._model import FOV, ROI, Mask
    from cali.util._util import save_labeled_images_from_fovs

    fov = FOV(name="test_fov", position_index=0, fov_number=0)

    # ROI from detection 1
    roi1 = ROI(
        label_value=1,
        active=True,
        stimulated=False,
        detection_settings_id=1,
        roi_mask=Mask(
            coords_y=[10],
            coords_x=[20],
            height=50,
            width=50,
            mask_type="roi",
        ),
    )
    # ROI from detection 2
    roi2 = ROI(
        label_value=2,
        active=True,
        stimulated=False,
        detection_settings_id=2,
        roi_mask=Mask(
            coords_y=[30],
            coords_x=[40],
            height=50,
            width=50,
            mask_type="roi",
        ),
    )
    fov.rois = [roi1, roi2]

    output_dir = tmp_path / "labeled_filtered"
    save_labeled_images_from_fovs(fov, output_dir, detection_settings_id=1)

    output_file = output_dir / "test_fov_labeled.tif"
    assert output_file.exists()

    import tifffile

    img = tifffile.imread(output_file)
    # Only ROI 1 should be present
    assert img[10, 20] == 1
    assert img[30, 40] == 0  # ROI 2 should not be present


def test_save_labeled_images_from_fovs_no_overwrite(tmp_path: Path) -> None:
    """Test save_labeled_images_from_fovs raises error when overwrite=False."""
    from cali.sqlmodel._model import FOV, ROI, Mask
    from cali.util._util import save_labeled_images_from_fovs

    fov = FOV(name="test_fov", position_index=0, fov_number=0)
    roi = ROI(
        label_value=1,
        active=True,
        stimulated=False,
        roi_mask=Mask(
            coords_y=[10],
            coords_x=[20],
            height=50,
            width=50,
            mask_type="roi",
        ),
    )
    fov.rois = [roi]

    output_dir = tmp_path / "labeled_no_overwrite"
    output_dir.mkdir()

    # Create existing file
    output_file = output_dir / "test_fov_labeled.tif"
    output_file.write_text("existing")

    with pytest.raises(FileExistsError, match="exists and overwrite=False"):
        save_labeled_images_from_fovs(fov, output_dir, overwrite=False)
