"""Test that wizard-created plate plans work correctly with data_to_plate."""

import tempfile
from pathlib import Path

import useq
from useq import register_well_plates

# Get the test data directory relative to this file
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_wizard_plate_plan_with_data_positions() -> None:
    """Test that wizard-created plate plans map to actual data positions."""
    # Register custom plate
    register_well_plates(
        {
            "dish-35mm-round": {
                "rows": 1,
                "columns": 1,
                "well_spacing": 0.0,
                "well_size": 35.0,
                "circular_wells": True,
                "name": "dish-35mm-round",
            },
        }
    )

    from cali.sqlmodel import Experiment
    from cali.sqlmodel._data_to_plate import data_to_plate

    # Simulate wizard-created plate plan
    wizard_plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("dish-35mm-round"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0, 0),),  # This format causes duplicate wells bug
        well_points_plan=useq.RandomPoints(num_points=6),
    )

    # Simulate data with positions (like from MM-GUI without HCS)
    data_path = str(TEST_DATA_DIR / "no_hcs" / "no_hcs.tensorstore.zarr")

    # Create experiment
    experiment = Experiment(name="Test", description="Test")

    # Call data_to_plate
    plate = data_to_plate(data_path, experiment, plate_plan=wizard_plate_plan)

    # Verify results
    assert plate is not None, "Plate should be created"
    assert len(plate.wells) == 1, f"Expected 1 well, got {len(plate.wells)}"
    assert plate.wells[0].name == "A1", f"Expected well A1, got {plate.wells[0].name}"
    assert len(plate.wells[0].fovs) == 2, (
        f"Expected 6 FOVs, got {len(plate.wells[0].fovs)}"
    )

    # Verify FOV names are correct
    fov_names = [fov.name for fov in plate.wells[0].fovs]
    expected_names = [f"A1_{i:04d}" for i in range(2)]
    assert fov_names == expected_names, (
        f"FOV names mismatch: {fov_names} != {expected_names}"
    )

    # Verify position indices are sequential
    position_indices = [fov.position_index for fov in plate.wells[0].fovs]
    assert position_indices == list(range(2)), (
        f"Position indices should be 0-1, got {position_indices}"
    )


def test_wizard_plate_plan_database_creation() -> None:
    """Test that wizard-created plate plans create correct database structure."""
    from cali.sqlmodel import (
        Experiment,
        load_experiment_from_database,
        save_experiment_to_database,
    )
    from cali.sqlmodel._data_to_plate import data_to_plate

    # Register custom plate
    register_well_plates(
        {
            "dish-35mm-round": {
                "rows": 1,
                "columns": 1,
                "well_spacing": 0.0,
                "well_size": 35.0,
                "circular_wells": True,
                "name": "dish-35mm-round",
            },
        }
    )

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.cali"

        # Simulate wizard-created plate plan
        wizard_plate_plan = useq.WellPlatePlan(
            plate=useq.WellPlate.from_str("dish-35mm-round"),
            a1_center_xy=(0.0, 0.0),
            selected_wells=((0, 0),),
            well_points_plan=useq.RandomPoints(num_points=6),
        )

        # Create experiment and populate with data
        # Use no_hcs data which doesn't have embedded plate metadata,
        # so the wizard's plate plan will actually be applied
        data_path = str(TEST_DATA_DIR / "no_hcs" / "no_hcs.tensorstore.zarr")
        experiment = Experiment(name="Test Wizard", description="Test")
        data_to_plate(data_path, experiment, plate_plan=wizard_plate_plan)

        # Save to database
        save_experiment_to_database(experiment, tmpdir, database_name="test.cali")

        # Load back from database
        loaded_exp = load_experiment_from_database(db_path)

        # Verify database structure
        assert loaded_exp is not None, "Failed to load experiment from database"
        assert loaded_exp.plate is not None, "Experiment should have a plate"

        # Since no_hcs data has no plate metadata, the wizard's dish-35mm-round
        # plate plan should be used, creating 1 well (A1) with 2 FOVs
        assert len(loaded_exp.plate.wells) == 1, (
            f"Expected 1 well (from wizard plan), got {len(loaded_exp.plate.wells)}"
        )

        well = loaded_exp.plate.wells[0]
        assert well.name == "A1", f"Expected well A1, got {well.name}"
        assert len(well.fovs) == 2, f"Expected 2 FOVs, got {len(well.fovs)}"

        # Verify FOV structure matches wizard mapping
        for i, fov in enumerate(well.fovs):
            assert fov.name == f"A1_{i:04d}", (
                f"FOV {i}: Expected name A1_{i:04d}, got {fov.name}"
            )
            assert fov.position_index == i, (
                f"FOV {i}: Expected position_index {i}, got {fov.position_index}"
            )


if __name__ == "__main__":
    test_wizard_plate_plan_with_data_positions()
    print("✅ Database test passed!")
    print("Run with pytest to test GUI behavior")
