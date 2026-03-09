"""Test for plate_map_hash functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import useq
from sqlmodel import Session, create_engine
from useq import register_well_plates

from cali.sqlmodel import Experiment
from cali.sqlmodel._plate_map_util import compute_plate_map_hash

TEST_DATA_DIR = Path("tests/test_data")


def test_compute_plate_map_hash_none() -> None:
    """Test that compute_plate_map_hash returns None for None input."""
    assert compute_plate_map_hash(None) is None


def test_compute_plate_map_hash_stable() -> None:
    """Test that same plate_maps produce same hash."""
    plate_maps1 = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle", "A2": "Drug"},
    }
    plate_maps2 = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle", "A2": "Drug"},
    }

    hash1 = compute_plate_map_hash(plate_maps1)
    hash2 = compute_plate_map_hash(plate_maps2)

    assert hash1 == hash2
    assert hash1 is not None


def test_compute_plate_map_hash_order_independent() -> None:
    """Test that hash is independent of dict iteration order."""
    # Different insertion order
    plate_maps1 = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle"},
    }
    plate_maps2 = {
        "treatment": {"A1": "Vehicle"},
        "genotype": {"A2": "KO", "A1": "WT"},  # Different well order too
    }

    hash1 = compute_plate_map_hash(plate_maps1)
    hash2 = compute_plate_map_hash(plate_maps2)

    assert hash1 == hash2


def test_compute_plate_map_hash_different_values() -> None:
    """Test that different plate_maps produce different hashes."""
    plate_maps1 = {
        "genotype": {"A1": "WT", "A2": "KO"},
    }
    plate_maps2 = {
        "genotype": {"A1": "KO", "A2": "WT"},  # Swapped values
    }

    hash1 = compute_plate_map_hash(plate_maps1)
    hash2 = compute_plate_map_hash(plate_maps2)

    assert hash1 != hash2


# ============================================================================
# Plate Map Run Differentiation Tests
# ============================================================================


def test_plate_map_hash_changes_when_treatment_cleared() -> None:
    """Test that plate_map_hash changes when treatment is removed.

    This tests the core behavior: when a user runs with both genotype and treatment,
    then clears treatment and runs again, the hash should change.
    """
    # Simulate first run: both genotype and treatment
    plate_maps_both = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle", "A2": "Drug"},
    }

    hash1 = compute_plate_map_hash(plate_maps_both)

    # Simulate second run: only genotype (treatment cleared)
    plate_maps_genotype_only = {
        "genotype": {"A1": "WT", "A2": "KO"},
    }

    hash2 = compute_plate_map_hash(plate_maps_genotype_only)

    # The hashes MUST be different
    assert hash1 is not None, "Hash with both conditions should not be None"
    assert hash2 is not None, "Hash with genotype only should not be None"
    assert hash1 != hash2, (
        f"Plate map hashes should differ when treatment is removed!\n"
        f"Both conditions hash: {hash1}\n"
        f"Genotype only hash: {hash2}"
    )


def test_plate_map_hash_changes_when_genotype_cleared() -> None:
    """Test that plate_map_hash changes when genotype is removed."""
    # Simulate first run: both genotype and treatment
    plate_maps_both = {
        "genotype": {"A1": "WT"},
        "treatment": {"A1": "Vehicle"},
    }

    hash1 = compute_plate_map_hash(plate_maps_both)

    # Simulate second run: only treatment (genotype cleared)
    plate_maps_treatment_only = {
        "treatment": {"A1": "Vehicle"},
    }

    hash2 = compute_plate_map_hash(plate_maps_treatment_only)

    # The hashes MUST be different
    assert hash1 is not None
    assert hash2 is not None
    assert hash1 != hash2, (
        f"Plate map hashes should differ when genotype is removed!\n"
        f"Both conditions hash: {hash1}\n"
        f"Treatment only hash: {hash2}"
    )


def test_plate_map_hash_none_when_empty() -> None:
    """Test that plate_map_hash is None when plate_maps is None."""
    assert compute_plate_map_hash(None) is None


def test_plate_map_hash_stable_for_same_config() -> None:
    """Test that the same plate_maps configuration produces the same hash."""
    plate_maps = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle", "A2": "Drug"},
    }

    hash1 = compute_plate_map_hash(plate_maps)
    hash2 = compute_plate_map_hash(plate_maps)

    assert hash1 == hash2, "Same plate_maps should produce same hash"


def test_plate_maps_stored_in_plate(tmp_path: Path) -> None:
    """Integration test: verify plate_maps are stored in Plate and can be updated.

    This tests that plate_maps are a property of the plate, not the analysis run.
    """
    from cali.sqlmodel import save_experiment_to_database

    # Create test database
    db_path = tmp_path / "test.cali"

    # Create experiment with initial plate_maps (both genotype and treatment)
    experiment = Experiment.create_from_data(
        name="plate_map_test",
        data_path=Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"),
        plate_maps={
            "genotype": {"B5": "WT", "B6": "KO"},
            "treatment": {"B5": "Vehicle", "B6": "Drug"},
        },
    )

    # Save to database
    save_experiment_to_database(experiment, tmp_path, database_name=db_path.name)

    # Verify plate_maps saved correctly
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        exp = session.get(Experiment, 1)
        assert exp is not None
        assert exp.plate is not None
        assert exp.plate.plate_maps == {
            "genotype": {"B5": "WT", "B6": "KO"},
            "treatment": {"B5": "Vehicle", "B6": "Drug"},
        }
    engine.dispose()

    # Now update plate_maps to only genotype (simulating user clearing treatment)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        exp = session.get(Experiment, 1)
        assert exp is not None
        assert exp.plate is not None

        # Update to only genotype
        exp.plate.plate_maps = {
            "genotype": {"B5": "WT", "B6": "KO"},
        }
        session.commit()
    engine.dispose()

    # Verify the update persisted
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        exp = session.get(Experiment, 1)
        assert exp is not None
        assert exp.plate is not None
        assert exp.plate.plate_maps == {
            "genotype": {"B5": "WT", "B6": "KO"},
        }, "Plate maps should be updated to only genotype"
    engine.dispose()


# ============================================================================
# Wizard Plate Plan Tests
# ============================================================================


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
