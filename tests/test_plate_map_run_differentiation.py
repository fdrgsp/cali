"""Test that different plate map configurations create different hashes."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session, create_engine

from cali.sqlmodel import Experiment
from cali.sqlmodel._plate_map_util import compute_plate_map_hash


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
        data_path=Path("tests/test_data/test_for_plot/evk.tensorstore.zarr"),
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
