"""Test that different plate map configurations create different hashes and runs."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali.sqlmodel import CaliResult, Experiment
from cali.sqlmodel._plate_map_util import compute_plate_map_hash


def test_plate_map_hash_changes_when_treatment_cleared() -> None:
    """Test that plate_map_hash changes when treatment is removed.

    This tests the core bug: when a user runs with both genotype and treatment,
    then clears treatment and runs again, the hash should change to create a new run.
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


def test_plate_map_changes_create_new_runs(tmp_path: Path) -> None:
    """Integration test: verify changing plate maps creates separate runs.

    This is the critical regression test for the bug where changing plate maps
    (e.g., clearing treatment) didn't create a new run.
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

        # Simulate first run by creating a result with both conditions
        from cali.sqlmodel import (
            AnalysisSettings,
            DetectionSettings,
            ExtractionSettings,
        )

        det_settings = DetectionSettings(method="cellpose", model_type="cpsam")
        ext_settings = ExtractionSettings(dff_window=200)
        ana_settings = AnalysisSettings(experiment_type="Evoked Activity")

        session.add(det_settings)
        session.add(ext_settings)
        session.add(ana_settings)
        session.commit()

        # Create result for run 1 (both genotype and treatment)
        plate_maps_both = exp.plate.plate_maps
        hash_both = compute_plate_map_hash(plate_maps_both)

        result1 = CaliResult(
            experiment=1,
            detection_settings_id=det_settings.id,
            extraction_settings_id=ext_settings.id,
            analysis_settings_id=ana_settings.id,
            positions_detected=[0, 1],
            positions_extracted=[0, 1],
            positions_analyzed=[0, 1],
            plate_maps=plate_maps_both,
            plate_map_hash=hash_both,
        )
        session.add(result1)
        session.commit()
        result1_id = result1.id  # Save ID before detaching
    engine.dispose()

    # Now change plate_maps to only genotype (simulating user clearing treatment)
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

    # Verify that a query for existing results with the new plate_map_hash
    # does NOT find the old result
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        exp = session.get(Experiment, 1)
        assert exp is not None
        assert exp.plate is not None
        new_plate_maps = exp.plate.plate_maps
        new_hash = compute_plate_map_hash(new_plate_maps)

        # Verify hashes are different
        assert new_hash != hash_both, (
            "Different plate_maps should have different hashes"
        )

        # Query for results with new hash (should find nothing)
        matching_results = session.exec(
            select(CaliResult).where(
                CaliResult.experiment == 1,
                CaliResult.detection_settings_id == 1,
                CaliResult.extraction_settings_id == 1,
                CaliResult.analysis_settings_id == 1,
                CaliResult.plate_map_hash == new_hash,
            )
        ).all()

        assert len(matching_results) == 0, (
            "Should find no results with new plate_map_hash "
            "(proving a new run would be created)"
        )

        # Query for results with old hash (should find result1)
        old_results = session.exec(
            select(CaliResult).where(
                CaliResult.experiment == 1,
                CaliResult.detection_settings_id == 1,
                CaliResult.extraction_settings_id == 1,
                CaliResult.analysis_settings_id == 1,
                CaliResult.plate_map_hash == hash_both,
            )
        ).all()

        assert len(old_results) == 1, (
            "Should still find the original result with old hash"
        )
        assert old_results[0].id == result1_id
    engine.dispose()
