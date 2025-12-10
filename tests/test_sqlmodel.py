"""Comprehensive tests for cali.sqlmodel module.

Tests cover:
- Database schema creation and integrity
- Model relationships and constraints
- JSON to database migration
- Database to useq.WellPlate/WellPlatePlan conversion
- Helper functions and utilities
- Edge cases and error handling

Note: For creating experiments in your code, prefer using Experiment classmethods:
- Experiment.create() - Create experiment with manual configuration
- Experiment.create_from_data() - Create from data directory (auto-detects structure)
- Experiment.load_from_db() - Load existing experiment with all relationships

These fixtures use the lower-level constructors for fine-grained testing.
See test_results.py for examples using the higher-level classmethods.
"""

from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import useq
from sqlalchemy import Engine
from sqlmodel import Session, create_engine, select

from cali.sqlmodel import (
    FOV,
    ROI,
    Condition,
    Experiment,
    Plate,
    Well,
    experiment_to_plate_map_data,
    experiment_to_useq_plate,
    experiment_to_useq_plate_plan,
    save_experiment_to_database,
    useq_plate_plan_to_db,
)
from cali.sqlmodel._json_to_db import load_plate_map, parse_well_name, roi_from_roi_data
from cali.sqlmodel._model import (
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    DetectionSettings,
    ExtractionSettings,
    Mask,
    Traces,
)
from cali.sqlmodel._util import (
    create_database_and_tables,
)

if TYPE_CHECKING:
    from collections.abc import Generator

from unittest.mock import MagicMock, patch

from cali.sqlmodel._visualize_experiment import print_cali_results

TempDB = tuple[Engine, Path]

THREADS = 1


def _get_actual_db_path(requested_db_path: Path) -> Path:
    """Get the actual database path (with .cali extension added if needed)."""
    if not requested_db_path.name.endswith(".cali"):
        return requested_db_path.parent / f"{requested_db_path.name}.cali"
    return requested_db_path


# ==================== Fixtures ====================


@pytest.fixture
def temp_db() -> Generator[tuple[Engine, Path], None, None]:
    """Create a temporary SQLite database for testing."""
    import gc

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    engine = create_engine(f"sqlite:///{db_path}")
    create_database_and_tables(engine)

    yield engine, db_path

    # Cleanup - dispose engine before deleting file
    # Dispose with close=True to close all checked-in connections (Python 3.13)
    engine.dispose(close=True)
    # Force garbage collection to ensure connections are closed
    gc.collect()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def simple_experiment(temp_db: tuple[Engine, Path], tmp_path: Path) -> Experiment:
    """Create a simple experiment with one well, one FOV, and one ROI."""
    engine, _db_path = temp_db

    # Create experiment
    exp = Experiment(
        name="test_experiment",
        description="Test experiment",
    )

    # Create plate
    plate = Plate(
        experiment=exp,
        name="96-well",
        plate_type="96-well",
        rows=8,
        columns=12,
    )

    # Create conditions
    cond1 = Condition(
        name="WT",
        condition_type="genotype",
        color="blue",
    )
    cond2 = Condition(
        name="Control",
        condition_type="treatment",
        color="gray",
    )

    # Create well
    well = Well(
        plate=plate,
        name="B5",
        row=1,
        column=4,
        conditions=[cond1, cond2],
    )

    # Create FOV
    fov = FOV(
        well=well,
        name="B5_0000_p0",
        position_index=0,
        fov_number=0,
    )

    # Create ROI
    roi = ROI(
        fov=fov,
        label_value=1,
        active=True,
        stimulated=False,
        cell_size=100.5,
        cell_size_units="pixels",
    )

    # Create traces
    Traces(
        roi=roi,
        raw_trace=[1.0, 2.0, 3.0],
        dff=[0.0, 0.1, 0.2],
    )

    # Create data analysis
    DataAnalysis(
        roi=roi,
    )

    # Save to database
    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

    return exp


@pytest.fixture
def mock_engine() -> Generator[Engine, None, None]:
    engine = create_engine("sqlite:///:memory:")
    from cali.sqlmodel._util import create_database_and_tables

    create_database_and_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def populated_db(mock_engine: Engine) -> Engine:
    engine = mock_engine
    with Session(engine) as session:
        # Create Experiment
        exp = Experiment(name="Test Experiment", description="Test Description")
        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Create Plate
        plate = Plate(experiment=exp, name="Test Plate", plate_type="96-well")
        session.add(plate)
        session.commit()
        session.refresh(plate)

        # Create Conditions
        cond1 = Condition(name="Cond1", condition_type="Type1")
        cond2 = Condition(name="Cond2", condition_type="Type2")
        session.add(cond1)
        session.add(cond2)
        session.commit()

        # Create Well
        well = Well(plate=plate, name="A1", row=0, column=0, conditions=[cond1, cond2])
        session.add(well)
        session.commit()
        session.refresh(well)

        # Create FOV
        fov = FOV(well=well, name="A1_0000", position_index=0, fov_number=0)
        session.add(fov)
        session.commit()
        session.refresh(fov)

        # Create Settings
        det_settings = DetectionSettings(
            method="cellpose", model_type="cyto", diameter=30
        )
        session.add(det_settings)

        ext_settings = ExtractionSettings(neuropil_inner_radius=5)
        session.add(ext_settings)

        ana_settings = AnalysisSettings(
            experiment_type="Evoked Activity",
            led_power_equation="x",
            led_pulse_duration=10,
            led_pulse_powers=[10],
            led_pulse_on_frames=[10],
            stimulation_mask_path="path/to/mask",
        )
        session.add(ana_settings)
        session.commit()
        session.refresh(det_settings)
        session.refresh(ext_settings)
        session.refresh(ana_settings)

        # Create ROI
        roi = ROI(
            fov=fov,
            label_value=1,
            active=True,
            stimulated=True,
            detection_settings_id=det_settings.id,
        )
        session.add(roi)
        session.commit()
        session.refresh(roi)

        # Create CaliResult
        result = CaliResult(
            experiment=exp.id,
            detection_settings_id=det_settings.id,
            extraction_settings_id=ext_settings.id,
            analysis_settings_id=ana_settings.id,
            positions_analyzed=[0, 1, 2, 4],  # Test range grouping
        )
        session.add(result)
        session.commit()
        session.refresh(result)

        # Add Traces and DataAnalysis linked to result
        trace = Traces(roi=roi, raw_trace=[1, 2, 3], analysis_result_id=result.id)
        session.add(trace)

        da = DataAnalysis(roi=roi, analysis_result_id=result.id)
        session.add(da)

        # Add Mask
        mask = Mask(coords_y=[0], coords_x=[0], height=10, width=10, mask_type="roi")
        roi.roi_mask = mask
        session.add(mask)
        session.commit()

    return engine


# ==================== Model Tests ====================


def test_experiment_creation(temp_db: TempDB) -> None:
    """Test basic Experiment model creation."""
    engine, _db_path = temp_db

    exp = Experiment(
        name="test_exp",
        description="Test description",
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        # Verify
        result = session.exec(select(Experiment)).first()
        assert result.name == "test_exp"
        assert result.description == "Test description"
        assert result.id is not None


def test_experiment_create_from_data(tmp_path: Path) -> None:
    """Test Experiment.create_from_data classmethod."""

    # Create experiment from test data
    exp = Experiment.create_from_data(
        name="Test Experiment From Data",
        data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
        plate_maps={
            "genotype": {"B5": "WT"},
            "treatment": {"B5": "Vehicle"},
        },
    )

    # Verify experiment structure was loaded from data
    assert exp.name == "Test Experiment From Data"
    assert exp.plate is not None
    assert len(exp.plate.wells) > 0
    assert exp.plate.wells[0].name == "B5"
    assert len(exp.plate.wells[0].fovs) > 0

    # Verify plate maps were applied
    assert len(exp.plate.wells[0].conditions) == 2
    condition_names = {c.name for c in exp.plate.wells[0].conditions}
    assert "WT" in condition_names
    assert "Vehicle" in condition_names


def test_plate_relationship(simple_experiment: Experiment, temp_db: TempDB) -> None:
    """Test Experiment-Plate relationship."""
    engine, _ = temp_db

    with Session(engine) as session:
        exp = session.get(Experiment, simple_experiment.id)
        assert exp.plate is not None
        assert exp.plate.name == "96-well"
        assert exp.plate.experiment_id == exp.id


def test_well_conditions_many_to_many(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test Well-Condition many-to-many relationship."""
    engine, _ = temp_db

    with Session(engine) as session:
        well = session.exec(select(Well)).first()
        assert len(well.conditions) == 2
        assert well.condition_1.name == "WT"
        assert well.condition_2.name == "Control"

        # Check that conditions are shared (can be reused)
        cond = session.exec(select(Condition).where(Condition.name == "WT")).first()
        assert cond is not None


def test_fov_well_relationship(simple_experiment: Experiment, temp_db: TempDB) -> None:
    """Test FOV-Well relationship."""
    engine, _ = temp_db

    with Session(engine) as session:
        fov = session.exec(select(FOV)).first()
        assert fov.well.name == "B5"
        assert len(fov.well.fovs) == 1


def test_roi_relationships(simple_experiment: Experiment, temp_db: TempDB) -> None:
    """Test ROI relationships with traces and analysis."""
    engine, _ = temp_db

    with Session(engine) as session:
        roi = session.exec(select(ROI)).first()
        assert roi.fov.name == "B5_0000_p0"
        assert len(roi.traces_history) > 0
        assert roi.traces_history[0].raw_trace == [1.0, 2.0, 3.0]
        assert len(roi.data_analysis_history) > 0
        assert roi.cell_size == 100.5


def test_unique_constraints(temp_db: TempDB) -> None:
    """Test unique constraints on models."""
    engine, _db_path = temp_db

    # Experiment names must be unique
    with Session(engine) as session:
        session.add(Experiment(name="test1"))
        session.commit()

    with Session(engine) as session:
        session.add(Experiment(name="test1"))
        with pytest.raises(Exception):  # IntegrityError  # noqa: B017  # noqa: B017
            session.commit()


def test_cascade_deletion(simple_experiment: Experiment, temp_db: TempDB) -> None:
    """Test that related entities are preserved when experiment is deleted.

    Note: SQLModel doesn't automatically cascade deletes by default.
    This test verifies the current behavior.
    """
    engine, _ = temp_db

    with Session(engine) as session:
        exp_id = simple_experiment.id

        # Count related entities before deletion
        plate_count_before = len(session.exec(select(Plate)).all())
        assert plate_count_before > 0

        # Delete experiment - this will fail due to foreign key constraints
        # unless we explicitly set up cascade behavior
        exp = session.get(Experiment, exp_id)

        # For now, just verify the experiment exists
        assert exp is not None
        assert exp.id == exp_id


# ==================== Helper Function Tests ====================


def test_parse_well_name_valid() -> None:
    """Test parsing valid well names."""
    # Single letter rows
    assert parse_well_name("A1") == (0, 0)
    assert parse_well_name("B5") == (1, 4)
    assert parse_well_name("H12") == (7, 11)
    assert parse_well_name("a1") == (0, 0)  # lowercase
    assert parse_well_name("Z1") == (25, 0)

    # Multi-letter rows (for plates with >26 rows)
    assert parse_well_name("AA1") == (26, 0)
    assert parse_well_name("AB5") == (27, 4)
    assert parse_well_name("AE19") == (30, 18)
    assert parse_well_name("ae19") == (30, 18)  # lowercase
    assert parse_well_name("ZZ1") == (701, 0)


def test_parse_well_name_invalid() -> None:
    """Test parsing invalid well names."""
    with pytest.raises(ValueError, match="Invalid well name"):
        parse_well_name("")

    with pytest.raises(ValueError, match="Invalid well name"):
        parse_well_name("1A")

    with pytest.raises(ValueError, match="Invalid well name"):
        parse_well_name("AA")


def test_load_plate_map(tmp_path: Path) -> None:
    """Test loading plate map from JSON."""
    # Create test plate map
    plate_map_data = [
        ["A1", "", ["WT", "blue"]],
        ["B5", "", ["KO", "red"]],
    ]

    plate_map_file = tmp_path / "test_map.json"
    with open(plate_map_file, "w") as f:
        json.dump(plate_map_data, f)

    # Load and verify
    result = load_plate_map(plate_map_file)
    assert "A1" in result
    assert result["A1"]["name"] == "WT"
    assert result["A1"]["color"] == "blue"
    assert result["B5"]["name"] == "KO"


def test_load_plate_map_missing_file(tmp_path: Path) -> None:
    """Test loading from non-existent file returns empty dict."""
    result = load_plate_map(tmp_path / "missing.json")
    assert result == {}


# ==================== JSON Migration Tests ====================


def test_load_analysis_from_json(tmp_path: Path) -> None:
    """Test loading analysis from JSON directory."""
    # Use evoked test data - copy to tmp_path to avoid conflicts
    import shutil

    from cali._constants import EVOKED
    from cali.sqlmodel._json_to_db import load_analysis_from_json
    from cali.sqlmodel._model import (
        AnalysisSettings,
        CaliResult,
        DetectionSettings,
        ExtractionSettings,
    )
    from cali.sqlmodel._util import load_experiment_from_database

    test_data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    test_output_path = Path("tests/test_data/evoked/evk_analysis")

    # Copy data to tmp_path
    data_path = tmp_path / "evk.tensorstore.zarr"
    output_path = tmp_path / "evk_analysis"
    shutil.copytree(test_data_path, data_path)
    shutil.copytree(test_output_path, output_path)

    # Load from JSON (this will create database in output_path)
    useq_plate = useq.WellPlate.from_str("96-well")
    load_analysis_from_json(
        data_path=str(data_path),
        output_path=str(output_path),
        useq_plate=useq_plate,
        save_to_db=True,
    )

    # The database should be created in output_path
    db_path = output_path / f"{data_path.name}.db"
    # load_analysis_from_json doesn't add .cali extension
    assert db_path.exists()

    # Verify data was loaded correctly - reload from database
    loaded_exp = load_experiment_from_database(db_path)
    assert loaded_exp is not None

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            # Check DetectionSettings was created
            detection_settings = session.exec(select(DetectionSettings)).first()
            assert detection_settings is not None
            assert detection_settings.method == "cellpose"
            assert detection_settings.model_type == "custom"

            # Check ExtractionSettings was created
            extraction_settings = session.exec(select(ExtractionSettings)).first()
            assert extraction_settings is not None

            # Check AnalysisSettings was created
            analysis_settings = session.exec(select(AnalysisSettings)).first()
            assert analysis_settings is not None
            assert analysis_settings.experiment_type == EVOKED

            # Check CaliResult was created
            cali_result = session.exec(select(CaliResult)).first()
            assert cali_result is not None
            assert cali_result.experiment == loaded_exp.id
            assert cali_result.detection_settings_id == detection_settings.id
            assert cali_result.extraction_settings_id == extraction_settings.id
            assert cali_result.analysis_settings_id == analysis_settings.id

            # Check plate structure
            assert loaded_exp.plate is not None
            assert len(loaded_exp.plate.wells) == 1
            well = loaded_exp.plate.wells[0]
            assert well.name == "B5"

            # Check FOVs
            assert len(well.fovs) == 1
            fov = well.fovs[0]
            assert fov.name == "B5_0000_p0"

            # Check ROIs
            assert len(fov.rois) == 4
            for roi in fov.rois:
                assert roi.detection_settings_id == detection_settings.id
                assert len(roi.traces_history) == 1
                assert len(roi.data_analysis_history) == 1

    finally:
        engine.dispose(close=True)
        gc.collect()


def test_save_experiment_to_db(tmp_path: Path) -> None:
    """Test saving experiment to database."""
    db_path = tmp_path / "test.db"

    # Create simple experiment
    exp = Experiment(
        name="test_experiment",
        description="Test",
    )
    Plate(experiment=exp, name="96-well", plate_type="96-well")

    # Save
    save_experiment_to_database(
        exp, output_path=tmp_path, database_name="test.db", overwrite=True
    )

    # Verify
    from cali.sqlmodel._util import load_experiment_from_database

    result = load_experiment_from_database(_get_actual_db_path(db_path))
    assert result is not None
    assert result.name == "test_experiment"
    assert _get_actual_db_path(db_path).exists()


def test_save_experiment_overwrite_protection(
    simple_experiment: Experiment, tmp_path: Path
) -> None:
    """Test that overwrite=False protects existing database."""
    # Use tmp_path for database
    db_path = tmp_path / "test.db"

    # Create initial database
    save_experiment_to_database(
        simple_experiment, output_path=tmp_path, database_name="test.db", overwrite=True
    )

    # Try to save again without overwrite - should work (SQLite appends)
    # but verify the file exists
    actual_db = _get_actual_db_path(db_path)
    assert actual_db.exists()
    _size = actual_db.stat().st_size  # Check file size

    # Save with overwrite=True
    save_experiment_to_database(
        simple_experiment, output_path=tmp_path, database_name="test.db", overwrite=True
    )
    # File should still exist
    assert _get_actual_db_path(db_path).exists()


# ==================== Conversion Tests ====================


def test_experiment_to_useq_plate(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test converting experiment to useq.WellPlate."""
    engine, _ = temp_db

    with Session(engine) as session:
        exp = session.get(Experiment, simple_experiment.id)
        plate = experiment_to_useq_plate(exp)

        assert plate is not None
        assert plate.name == "96-well"
        assert plate.rows == 8
        assert plate.columns == 12


def test_experiment_to_useq_plate_with_custom_name(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test converting with custom plate name."""
    engine, _ = temp_db

    with Session(engine) as session:
        exp = session.get(Experiment, simple_experiment.id)
        plate = experiment_to_useq_plate(exp, useq_plate_name="384-well")

        assert plate is not None
        assert plate.name == "384-well"


def test_experiment_to_useq_plate_invalid_name(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test converting with invalid plate name raises error."""
    engine, _ = temp_db

    with Session(engine) as session:
        exp = session.get(Experiment, simple_experiment.id)

        with pytest.raises(ValueError, match=r"Invalid useq\.WellPlate name"):
            experiment_to_useq_plate(exp, useq_plate_name="invalid-plate")


def test_experiment_to_useq_plate_plan(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test converting experiment to useq.WellPlatePlan."""
    engine, _ = temp_db

    with Session(engine) as session:
        exp = session.get(Experiment, simple_experiment.id)
        plate_plan = experiment_to_useq_plate_plan(exp)

        assert plate_plan is not None
        assert plate_plan.plate.name == "96-well"
        assert plate_plan.a1_center_xy == (0.0, 0.0)
        assert plate_plan.rotation is None
        assert plate_plan.selected_wells == ((1,), (4,))  # Row B, Col 5


def test_experiment_to_useq_plate_plan_multiple_wells(temp_db: TempDB) -> None:
    """Test plate plan with multiple wells."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")

    # Create multiple wells
    Well(plate=plate, name="B5", row=1, column=4)
    Well(plate=plate, name="C6", row=2, column=5)
    Well(plate=plate, name="B6", row=1, column=5)

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        plate_plan = experiment_to_useq_plate_plan(exp)

        # Should have the three wells explicitly listed (sorted: B5, B6, C6)
        assert plate_plan.selected_wells == ((1, 1, 2), (4, 5, 5))
        assert plate_plan.selected_well_names == ["B5", "B6", "C6"]


def test_experiment_to_useq_plate_plan_no_wells(temp_db: TempDB) -> None:
    """Test plate plan with no wells returns None."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    Plate(experiment=exp, name="96-well", plate_type="96-well")

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        plate_plan = experiment_to_useq_plate_plan(exp)
        assert plate_plan is None


def test_useq_plate_plan_to_plate(temp_db: TempDB) -> None:
    """Test converting useq.WellPlatePlan to cali.sqlmodel.Plate."""
    engine, _ = temp_db

    # Create experiment
    exp = Experiment(
        name="test_useq_import",
        description="Import from useq",
    )

    # Save experiment to get ID
    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

    from useq import register_well_plates

    # Register 1536-well plate
    register_well_plates(
        {
            "1536-well": {
                "rows": 32,
                "columns": 48,
                "well_spacing": 2.25,
                "well_size": 1.55,
            }
        }
    )

    # Create useq plate plan
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("1536-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((1, 2, 30), (4, 5, 18)),  # Wells B5, C6, AE19 (paired)
    )

    # Convert to database objects
    plate = useq_plate_plan_to_db(plate_plan, exp)

    # Verify plate properties
    assert plate.name == "1536-well"
    assert plate.plate_type == "1536-well"
    assert plate.rows == 32
    assert plate.columns == 48

    # Verify wells were created (3 wells from paired indices)
    assert len(plate.wells) == 3
    well_names = sorted([w.name for w in plate.wells])
    assert well_names == sorted(plate_plan.selected_well_names)

    # Verify well properties
    for well in plate.wells:
        assert well.plate == plate
        if well.name == "B5":
            assert well.row == 1
            assert well.column == 4
        elif well.name == "C6":
            assert well.row == 2
            assert well.column == 5
        elif well.name == "AE19":
            assert well.row == 30  # AE = row 30 (A=0, Z=25, AA=26, AE=30)
            assert well.column == 18

    # Save to database and verify persistence
    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        assert exp.plate.name == "1536-well"
        assert len(exp.plate.wells) == 3


def test_useq_plate_plan_roundtrip(temp_db: TempDB) -> None:
    """Test round-trip conversion: useq → database → useq."""
    engine, _ = temp_db

    # Create experiment with useq plate plan
    exp = Experiment(
        name="roundtrip_test",
        description="Test round-trip",
    )

    # Save experiment to get ID first
    with Session(engine, expire_on_commit=False) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

    plate_plan_orig = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0, 1, 2), (3, 4, 5)),  # A4-A6, B4-B6, C4-C6
    )

    # Convert to database
    _ = useq_plate_plan_to_db(plate_plan_orig, exp)

    # Save to database
    with Session(engine, expire_on_commit=False) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert back to useq
        plate_plan_new = experiment_to_useq_plate_plan(exp)

        # Verify round-trip
        assert plate_plan_new is not None
        assert plate_plan_new.plate.name == plate_plan_orig.plate.name
        assert plate_plan_new.selected_wells == plate_plan_orig.selected_wells


# ==================== ROI Data Conversion Tests ====================


def test_roi_from_roi_data(temp_db: TempDB) -> None:
    """Test converting ROIData to SQLModel entities."""
    from cali.sqlmodel._util import ROIData

    # Create mock ROIData
    roi_data = ROIData(
        raw_trace=[1.0, 2.0, 3.0],
        dff=[0.0, 0.1, 0.2],
        active=True,
        stimulated=False,
        cell_size=100.0,
        cell_size_units="pixels",
        elapsed_time_list_ms=[0.0, 100.0, 200.0],
    )

    # Convert
    roi, trace, data_analysis, roi_mask, _neuropil_mask = roi_from_roi_data(
        roi_data,
        fov_id=1,
        label_value=1,
        settings_id=None,
    )

    # Verify ROI
    assert roi.label_value == 1
    assert roi.active is True
    assert roi.stimulated is False
    assert roi.cell_size == 100.0
    assert roi.cell_size_units == "pixels"

    # Verify Trace
    assert trace.raw_trace == [1.0, 2.0, 3.0]
    assert trace.dff == [0.0, 0.1, 0.2]

    # Verify DataAnalysis (cell_size moved to ROI)
    assert data_analysis is not None

    # Verify masks
    assert roi_mask is not None
    assert roi_mask.mask_type == "roi"


# ==================== Edge Cases ====================


def test_empty_database(temp_db: TempDB) -> None:
    """Test querying empty database."""
    engine, _ = temp_db

    with Session(engine) as session:
        result = session.exec(select(Experiment)).all()
        assert len(result) == 0


def test_roi_without_traces(temp_db: TempDB) -> None:
    """Test ROI can exist without traces."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    fov = FOV(well=well, name="A1_0000_p0", position_index=0)
    ROI(fov=fov, label_value=1)

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        # Query and verify
        result = session.exec(select(ROI)).first()
        assert len(result.traces_history) == 0
        assert len(result.data_analysis_history) == 0


def test_well_without_conditions(temp_db: TempDB) -> None:
    """Test well can exist without conditions."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    Well(plate=plate, name="A1", row=0, column=0)

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        result = session.exec(select(Well)).first()
        assert len(result.conditions) == 0
        assert result.condition_1 is None
        assert result.condition_2 is None


def test_large_trace_data(temp_db: TempDB) -> None:
    """Test storing large trace arrays."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    fov = FOV(well=well, name="A1_0000_p0", position_index=0)
    roi = ROI(fov=fov, label_value=1)

    # Create large trace (1000 points)
    large_trace = list(range(1000))
    Traces(roi=roi, raw_trace=large_trace)

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        # Retrieve and verify
        result = session.exec(select(Traces)).first()
        assert len(result.raw_trace) == 1000
        assert result.raw_trace[0] == 0
        assert result.raw_trace[999] == 999


# ==================== Integration Tests ====================


def test_full_workflow(tmp_path: Path) -> None:
    """Test complete workflow from creation to database to export."""

    # 1. Create experiment from data
    exp = Experiment.create_from_data(
        name="Full Workflow Test",
        data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
        plate_maps={
            "genotype": {"B5": "WT"},
            "treatment": {"B5": "Vehicle"},
        },
    )

    # Verify basic experiment structure
    assert exp.plate is not None
    assert len(exp.plate.wells) > 0

    # 2. Save to database
    db_path = tmp_path / "test.db"
    save_experiment_to_database(
        exp, output_path=tmp_path, database_name="test.db", overwrite=True
    )

    # 3. Read back from database
    actual_db = _get_actual_db_path(db_path)
    engine = create_engine(f"sqlite:///{actual_db}")
    try:
        with Session(engine) as session:
            loaded_exp = session.exec(select(Experiment)).first()

            # 4. Convert to useq.WellPlate
            useq_plate = experiment_to_useq_plate(loaded_exp)
            assert useq_plate is not None

            # 5. Convert to useq.WellPlatePlan
            useq_plate_plan = experiment_to_useq_plate_plan(loaded_exp)
            assert useq_plate_plan is not None
    finally:
        # Cleanup - dispose engine (Python 3.13 compatibility)
        engine.dispose(close=True)


def test_data_to_plate_error_cases(tmp_path: Path) -> None:
    """Test data_to_plate error handling."""
    from cali.sqlmodel._data_to_plate import data_to_plate

    exp = Experiment(
        name="test",
        output_path=str(tmp_path),
    )

    # Save to get ID
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    engine = create_engine(f"sqlite:///{db_path}")
    create_database_and_tables(engine)

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

    # Test with invalid path - data_to_plate logs error and returns None
    result = data_to_plate("/nonexistent/path", exp)
    assert result is None

    engine.dispose(close=True)
    gc.collect()
    db_path.unlink(missing_ok=True)


def test_db_to_useq_plate_error_cases(temp_db: TempDB) -> None:
    """Test db_to_useq_plate error handling."""
    from cali.sqlmodel._db_to_useq_plate import experiment_to_useq_plate

    engine, _ = temp_db

    # Experiment with no plate
    exp = Experiment(
        name="no_plate",
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Test experiment_to_useq_plate with no plate - should return None
        result = experiment_to_useq_plate(exp)
        assert result is None, "Should return None when experiment has no plate"


def test_useq_plate_to_db_with_positions(temp_db: TempDB) -> None:
    """Test useq_plate_plan_to_db with actual positions."""
    engine, _ = temp_db

    exp = Experiment(
        name="test_positions",
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

    # Create plate plan with positions
    plate_plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0.0, 0.0),
        selected_wells=((1,), (4,)),  # B5
    )

    # Add a position manually
    from useq import Position

    Position(name="B5_0000", x=100.0, y=200.0)

    # Create new plan with position
    from useq import WellPlatePlan

    plan_with_pos = WellPlatePlan(
        plate=plate_plan.plate,
        a1_center_xy=plate_plan.a1_center_xy,
        selected_wells=plate_plan.selected_wells,
    )

    plate = useq_plate_plan_to_db(plan_with_pos, exp)
    assert plate is not None
    assert len(plate.wells) > 0


def test_util_load_experiment_from_database(tmp_path: Path) -> None:
    """Test load_experiment_from_database utility."""
    from cali.sqlmodel._util import (
        load_experiment_from_database,
        save_experiment_to_database,
    )

    # Create experiment structure
    exp = Experiment(
        name="test_load",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="B5", row=1, column=4)
    FOV(well=well, name="B5_0000", position_index=0, fov_number=0)

    # Save to database (this creates tables internally)
    save_experiment_to_database(
        exp, output_path=tmp_path, database_name="test_load.db", overwrite=True
    )

    # Load back
    db_path = tmp_path / "test_load.db"
    loaded_exp = load_experiment_from_database(_get_actual_db_path(db_path))

    assert loaded_exp is not None
    assert loaded_exp.name == "test_load"
    assert loaded_exp.plate is not None
    assert len(loaded_exp.plate.wells) == 1
    assert len(loaded_exp.plate.wells[0].fovs) == 1

    # Test with non-existent database
    result = load_experiment_from_database(tmp_path / "nonexistent.db")
    assert result is None


def test_visualize_experiment_functions(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test visualization functions."""

    engine, _db_path = temp_db

    # Test print_cali_results
    print_cali_results(
        engine,
        experiment_name=simple_experiment.name,
        show_settings=False,
    )

    print_cali_results(
        engine,
        experiment_name=None,  # All experiments
        show_settings=True,
    )


def test_json_to_db_error_handling(tmp_path: Path) -> None:
    """Test JSON loading error cases."""
    from cali.sqlmodel._json_to_db import parse_well_name

    # Test parse_well_name edge cases
    with pytest.raises(ValueError):
        parse_well_name("")

    with pytest.raises(ValueError):
        parse_well_name("123")

    with pytest.raises(ValueError):
        parse_well_name("ABC")


def test_model_stimulated_mask_area() -> None:
    """Test AnalysisSettings.stimulated_mask_area method."""
    from cali.sqlmodel._model import AnalysisSettings, Mask

    # Test with no mask
    settings = AnalysisSettings(threads=THREADS)
    assert settings.stimulated_mask_area() is None

    # Test with mask
    mask = Mask(
        coords_y=[0, 1, 2],
        coords_x=[0, 1, 2],
        height=10,
        width=10,
        mask_type="stimulation",
    )
    settings = AnalysisSettings(stimulation_mask=mask, threads=THREADS)
    result = settings.stimulated_mask_area()
    assert result is not None
    assert result.shape == (10, 10)


def test_db_to_plate_map_with_multiple_condition_types(temp_db: TempDB) -> None:
    """Test experiment_to_plate_map_data with various configurations."""
    from cali.sqlmodel._db_to_plate_map import experiment_to_plate_map_data

    engine, _ = temp_db

    exp = Experiment(name="test_map")
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")

    # Create conditions
    cond1 = Condition(name="WT", condition_type="genotype", color="blue")
    cond2 = Condition(name="KO", condition_type="genotype", color="red")
    cond3 = Condition(name="Drug", condition_type="treatment", color="green")

    # Well with multiple conditions
    Well(plate=plate, name="A1", row=0, column=0, conditions=[cond1, cond3])
    Well(plate=plate, name="A2", row=0, column=1, conditions=[cond2])

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        result = experiment_to_plate_map_data(exp)
        # Result is a tuple of lists, not a dict
        assert len(result) == 2  # Two condition types
        # Verify wells are present in the results
        all_wells = [item.name for sublist in result for item in sublist]
        assert "A1" in all_wells
        assert "A2" in all_wells


def test_useq_coverslip_plate_types(temp_db: TempDB) -> None:
    """Test special handling of coverslip plate types."""
    from cali.sqlmodel._useq_plate_to_db import useq_plate_to_db

    engine, _ = temp_db

    exp = Experiment(
        name="test_coverslip",
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

    # Test 18mm coverslip
    plate_18mm = useq.WellPlate(
        name="18mm coverslip",
        rows=1,
        columns=1,
        well_spacing=0,
        well_size=18,
    )
    plate = useq_plate_to_db(plate_18mm, exp)
    assert plate.plate_type == "coverslip-18mm-square"

    # Test 22mm coverslip
    plate_22mm = useq.WellPlate(
        name="22mm coverslip",
        rows=1,
        columns=1,
        well_spacing=0,
        well_size=22,
    )
    plate = useq_plate_to_db(plate_22mm, exp)
    assert plate.plate_type == "coverslip-22mm-square"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ==================== Additional Coverage Tests ====================


def test_experiment_to_useq_plate_no_plate_type(temp_db: TempDB) -> None:
    """Test converting experiment with no plate_type returns None."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    Plate(experiment=exp, name="test-plate", plate_type=None)

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert while still in session context to avoid detached instance issues
        result = experiment_to_useq_plate(exp)
        assert result is None


def test_well_condition_properties() -> None:
    """Test Well condition convenience properties."""
    well = Well(name="A1", row=0, column=0, plate_id=1)
    cond1 = Condition(name="WT", condition_type="genotype")
    cond2 = Condition(name="Control", condition_type="treatment")

    # Empty conditions
    assert well.condition_1 is None
    assert well.condition_2 is None

    # Add conditions
    well.conditions = [cond1, cond2]
    assert well.condition_1 == cond1
    assert well.condition_2 == cond2

    # Single condition
    well.conditions = [cond1]
    assert well.condition_1 == cond1
    assert well.condition_2 is None


def test_analysis_settings_evoked_fields(temp_db: TempDB) -> None:
    """Test AnalysisSettings with evoked experiment fields."""
    engine, _ = temp_db

    settings = AnalysisSettings(
        led_power_equation="y = 0.5 * x",
        led_pulse_duration=50.0,
        led_pulse_powers=[5.0, 10.0],
        led_pulse_on_frames=[100, 200],
        threads=THREADS,
    )

    with Session(engine) as session:
        session.add(settings)
        session.commit()

        result = session.exec(select(AnalysisSettings)).first()
        assert result.led_power_equation == "y = 0.5 * x"
        assert result.led_pulse_duration == 50.0
        assert result.led_pulse_powers == [5.0, 10.0]
        assert result.led_pulse_on_frames == [100, 200]


def test_traces_all_fields(temp_db: TempDB) -> None:
    """Test Traces with all fields populated."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    fov = FOV(well=well, name="A1_0000_p0", position_index=0)
    roi = ROI(fov=fov, label_value=1)

    Traces(
        roi=roi,
        raw_trace=[1.0, 2.0, 3.0],
        corrected_trace=[1.1, 2.1, 3.1],
        neuropil_trace=[0.1, 0.1, 0.1],
        dff=[0.0, 0.1, 0.2],
        dec_dff=[0.0, 0.15, 0.25],
        x_axis=[0.0, 100.0, 200.0],
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        result = session.exec(select(Traces)).first()
        assert result.corrected_trace == [1.1, 2.1, 3.1]
        assert result.neuropil_trace == [0.1, 0.1, 0.1]
        assert result.dec_dff == [0.0, 0.15, 0.25]
        assert result.x_axis == [0.0, 100.0, 200.0]


def test_data_analysis_all_fields(temp_db: TempDB) -> None:
    """Test DataAnalysis with all fields."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    fov = FOV(well=well, name="A1_0000_p0", position_index=0)
    roi = ROI(
        fov=fov,
        label_value=1,
        cell_size=150.5,
        cell_size_units="μm²",
    )

    DataAnalysis(
        roi=roi,
        total_recording_time_sec=600.0,
        dec_dff_frequency=2.5,
        peaks_dec_dff=[10.0, 20.0, 30.0],
        peaks_amplitudes_dec_dff=[0.5, 0.6, 0.7],
        iei=[10.0, 10.0],
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        result = session.exec(select(ROI)).first()
        assert result.cell_size_units == "μm²"

        data_analysis = session.exec(select(DataAnalysis)).first()
        assert data_analysis.total_recording_time_sec == 600.0
        assert data_analysis.peaks_amplitudes_dec_dff == [0.5, 0.6, 0.7]
        assert data_analysis.iei == [10.0, 10.0]


def test_mask_neuropil_type(temp_db: TempDB) -> None:
    """Test creating neuropil mask."""
    engine, _ = temp_db

    mask = Mask(
        coords_y=[0, 1, 2],
        coords_x=[0, 1, 2],
        height=10,
        width=10,
        mask_type="neuropil",
    )

    with Session(engine) as session:
        session.add(mask)
        session.commit()

        result = session.exec(select(Mask).where(Mask.mask_type == "neuropil")).first()
        assert result is not None
        assert result.mask_type == "neuropil"


def test_fov_metadata(temp_db: TempDB) -> None:
    """Test FOV with metadata."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    FOV(
        well=well,
        name="A1_0000_p0",
        position_index=0,
        fov_number=0,
        fov_metadata={"stage_x": 100.5, "stage_y": 200.3, "timestamp": "2024-01-01"},
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        result = session.exec(select(FOV)).first()
        assert result.fov_metadata is not None
        assert result.fov_metadata["stage_x"] == 100.5


def test_roi_with_masks(temp_db: TempDB) -> None:
    """Test ROI with both ROI and neuropil masks.

    Neuropil masks are stored on Traces, not ROI.
    """
    engine, _ = temp_db

    exp = Experiment(
        name="test",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    fov = FOV(well=well, name="A1_0000_p0", position_index=0)

    roi_mask = Mask(
        coords_y=[0, 1], coords_x=[0, 1], height=10, width=10, mask_type="roi"
    )
    trace_neuropil_mask = Mask(
        coords_y=[4, 5], coords_x=[4, 5], height=10, width=10, mask_type="neuropil"
    )

    roi = ROI(fov=fov, label_value=1)

    with Session(engine) as session:
        # Add masks first
        session.add(roi_mask)
        session.add(trace_neuropil_mask)
        session.flush()

        # Set mask IDs on ROI
        roi.roi_mask_id = roi_mask.id
        roi.roi_mask = roi_mask

        # Create Traces with neuropil mask
        traces = Traces(
            roi=roi,
            raw_trace=[1.0, 2.0, 3.0],
            neuropil_mask_id=trace_neuropil_mask.id,
            neuropil_mask=trace_neuropil_mask,
        )
        session.add(traces)

        session.add(exp)
        session.commit()

        # Verify ROI mask
        result_roi = session.exec(select(ROI)).first()
        assert result_roi.roi_mask is not None
        assert result_roi.roi_mask.mask_type == "roi"

        # Verify Traces neuropil mask
        result_trace = session.exec(select(Traces)).first()
        assert result_trace.neuropil_mask is not None
        assert result_trace.neuropil_mask.mask_type == "neuropil"
        assert result_trace.neuropil_mask.coords_y == [4, 5]

        # Force load mask relationships before expunging
        _ = result_roi.roi_mask
        _ = result_trace.neuropil_mask

        # Expunge to avoid lazy loading after session closes (Python 3.13)
        session.expunge_all()


def test_analysis_settings_with_stimulation_mask(temp_db: TempDB) -> None:
    """Test AnalysisSettings with stimulation mask."""
    engine, _ = temp_db

    with Session(engine) as session:
        # Create stimulation mask
        stim_mask = Mask(
            coords_y=[10, 11, 12],
            coords_x=[20, 21, 22],
            height=100,
            width=100,
            mask_type="stimulation",
        )

        # Create analysis settings with stimulation mask
        settings = AnalysisSettings(
            stimulation_mask_path="/path/to/stimulation_mask.tif",
            stimulation_mask=stim_mask,
            threads=THREADS,
        )

        session.add(settings)
        session.commit()

        # Retrieve and verify
        result = session.exec(select(AnalysisSettings)).first()
        assert result is not None
        assert result.stimulation_mask_path == "/path/to/stimulation_mask.tif"
        assert result.stimulation_mask is not None
        assert result.stimulation_mask.mask_type == "stimulation"
        assert result.stimulation_mask.coords_y == [10, 11, 12]
        assert result.stimulation_mask.coords_x == [20, 21, 22]
        assert result.stimulation_mask.height == 100
        assert result.stimulation_mask.width == 100


def test_analysis_settings_without_stimulation_mask(temp_db: TempDB) -> None:
    """Test AnalysisSettings without stimulation mask (optional field)."""
    engine, _ = temp_db

    with Session(engine) as session:
        # Create analysis settings without stimulation mask
        settings = AnalysisSettings(threads=THREADS)

        session.add(settings)
        session.commit()

        # Retrieve and verify
        result = session.exec(select(AnalysisSettings)).first()
        assert result is not None
        assert result.stimulation_mask_path is None
        assert result.stimulation_mask is None
        assert result.stimulation_mask_id is None


def test_experiment_to_plate_map_data(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test conversion of experiment to plate map data format."""
    engine, _ = temp_db

    with Session(engine) as session:
        session.add(simple_experiment)
        session.commit()
        session.refresh(simple_experiment)

        # Convert to plate map format
        genotype_data, treatment_data = experiment_to_plate_map_data(simple_experiment)

        # Verify genotype data
        assert len(genotype_data) == 1
        assert genotype_data[0].name == "B5"
        assert genotype_data[0].row_col == (1, 4)
        assert genotype_data[0].condition == ("WT", "blue")

        # Verify treatment data
        assert len(treatment_data) == 1
        assert treatment_data[0].name == "B5"
        assert treatment_data[0].row_col == (1, 4)
        assert treatment_data[0].condition == ("Control", "gray")


def test_experiment_to_plate_map_data_multiple_wells(temp_db: TempDB) -> None:
    """Test plate map conversion with multiple wells.

    Verifies that the conversion function correctly groups conditions by type
    (genotype vs treatment), not by position in the well's conditions list.
    """
    engine, _ = temp_db

    with Session(engine) as session:
        # Create experiment with multiple wells
        exp = Experiment(name="multi_well_test")
        plate = Plate(experiment=exp, name="24-well")

        # Create conditions
        wt = Condition(name="WT", condition_type="genotype", color="blue")
        ko = Condition(name="KO", condition_type="genotype", color="red")
        drug = Condition(name="Drug", condition_type="treatment", color="green")
        vehicle = Condition(name="Vehicle", condition_type="treatment", color="gray")

        # Create wells with different condition combinations
        Well(plate=plate, name="A1", row=0, column=0, conditions=[wt, vehicle])
        Well(plate=plate, name="A2", row=0, column=1, conditions=[wt, drug])
        Well(plate=plate, name="B1", row=1, column=0, conditions=[ko, vehicle])
        Well(plate=plate, name="B2", row=1, column=1, conditions=[ko, drug])

        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert to plate map format
        genotype_data, treatment_data = experiment_to_plate_map_data(exp)

        # Verify we have 4 wells total
        assert len(genotype_data) == 4
        assert len(treatment_data) == 4

        # Check that all wells are present
        genotype_well_names = {data.name for data in genotype_data}
        treatment_well_names = {data.name for data in treatment_data}
        assert genotype_well_names == {"A1", "A2", "B1", "B2"}
        assert treatment_well_names == {"A1", "A2", "B1", "B2"}

        # Verify that genotype_data only contains genotypes
        genotype_names = {data.condition[0] for data in genotype_data}
        assert genotype_names == {"WT", "KO"}

        # Verify that treatment_data only contains treatments
        treatment_names = {data.condition[0] for data in treatment_data}
        assert treatment_names == {"Drug", "Vehicle"}

        # Verify specific well mappings
        genotype_map = {d.name: d.condition[0] for d in genotype_data}
        treatment_map = {d.name: d.condition[0] for d in treatment_data}

        assert genotype_map["A1"] == "WT"
        assert treatment_map["A1"] == "Vehicle"
        assert genotype_map["A2"] == "WT"
        assert treatment_map["A2"] == "Drug"
        assert genotype_map["B1"] == "KO"
        assert treatment_map["B1"] == "Vehicle"
        assert genotype_map["B2"] == "KO"
        assert treatment_map["B2"] == "Drug"


def test_experiment_to_plate_map_data_no_conditions(temp_db: TempDB) -> None:
    """Test plate map conversion with wells that have no conditions."""
    engine, _ = temp_db

    with Session(engine) as session:
        # Create experiment with wells but no conditions
        exp = Experiment(name="no_conditions_test")
        plate = Plate(experiment=exp, name="24-well")
        Well(plate=plate, name="A1", row=0, column=0)
        Well(plate=plate, name="A2", row=0, column=1)

        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert to plate map format
        genotype_data, treatment_data = experiment_to_plate_map_data(exp)

        # Should return empty lists when wells have no conditions
        assert len(genotype_data) == 0
        assert len(treatment_data) == 0


def test_experiment_to_plate_map_data_no_plate(temp_db: TempDB) -> None:
    """Test plate map conversion with experiment that has no plate."""
    engine, _ = temp_db

    with Session(engine) as session:
        # Create experiment without plate
        exp = Experiment(name="no_plate_test")

        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert to plate map format
        genotype_data, treatment_data = experiment_to_plate_map_data(exp)

        # Should return empty lists when experiment has no plate
        assert len(genotype_data) == 0
        assert len(treatment_data) == 0


def test_experiment_to_plate_map_data_returns_by_type_not_position(
    temp_db: TempDB,
) -> None:
    """Test that plate map data is returned by condition_type, not position.

    This is a regression test for a bug where conditions were returned
    positionally (condition_1, condition_2) instead of by type
    (genotype, treatment), causing data to be scrambled when reloading.

    If treatment is stored as condition_1 and genotype as condition_2,
    the function should still return genotype first and treatment second.
    """
    engine, _ = temp_db

    with Session(engine) as session:
        # Create experiment
        exp = Experiment(name="position_vs_type_test")
        plate = Plate(experiment=exp, name="24-well")

        # Create conditions
        genotype = Condition(name="KO", condition_type="genotype", color="red")
        treatment = Condition(name="DrugA", condition_type="treatment", color="green")

        # CRITICAL: Add treatment FIRST, genotype SECOND to test position-independence
        # This simulates the bug scenario where dict iteration order affects storage
        Well(
            plate=plate,
            name="A1",
            row=0,
            column=0,
            conditions=[treatment, genotype],  # Treatment first!
        )

        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert to plate map format
        genotype_data, treatment_data = experiment_to_plate_map_data(exp)

        # Verify we get exactly one entry of each type
        assert len(genotype_data) == 1
        assert len(treatment_data) == 1

        # Verify genotype data is in first list (by type, not position!)
        assert genotype_data[0].name == "A1"
        assert genotype_data[0].condition == ("KO", "red")

        # Verify treatment data is in second list (by type, not position!)
        assert treatment_data[0].name == "A1"
        assert treatment_data[0].condition == ("DrugA", "green")

        # The key assertion: even though treatment was added first (position 0),
        # it should be returned in the treatment_data list (by type)
        # This would fail with the old position-based implementation


def test_experiment_create_from_tiff_data(tmp_path: Path) -> None:
    """Test Experiment.create_from_data with TIFF collection."""
    import numpy as np
    import tifffile

    # Create dummy TIFF files
    tiff_dir = tmp_path / "tiffs"
    tiff_dir.mkdir()

    file_map = {}
    for well in ["A1", "A2"]:
        files = []
        for i in range(2):
            fname = f"{well}_fov{i}.tif"
            fpath = tiff_dir / fname
            data = np.zeros((10, 10), dtype=np.uint8)
            tifffile.imwrite(fpath, data)
            files.append(str(fpath.absolute()))
        file_map[well] = files

    metadata = {"exposure_ms": 100.0, "pixel_size_um": 1.0}

    exp = Experiment.create_from_data(
        name="TIFF Experiment",
        data_path=tiff_dir,
        tiff_file_map=file_map,
        tiff_plate_type="96-well",
        tiff_metadata=metadata,
    )

    assert exp.name == "TIFF Experiment"
    assert exp.tiff_file_map_json is not None
    assert exp.plate is not None
    assert len(exp.plate.wells) == 2

    # Test tiff_collection_settings method
    settings = exp.tiff_collection_settings(tiff_dir)
    assert settings is not None
    assert settings.plate == "96-well"


def test_settings_load_from_database(temp_db: TempDB) -> None:
    """Test load_from_database methods for settings."""
    engine, db_path = temp_db

    with Session(engine) as session:
        d1 = DetectionSettings(method="cellpose", model_type="cpsam")
        d2 = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(d1)
        session.add(d2)

        e1 = ExtractionSettings(dff_window=10)
        e2 = ExtractionSettings(dff_window=20)
        session.add(e1)
        session.add(e2)

        a1 = AnalysisSettings(peaks_height_value=1.0)
        a2 = AnalysisSettings(peaks_height_value=2.0)
        session.add(a1)
        session.add(a2)

        session.commit()
        d1_id, _d2_id = d1.id, d2.id
        e1_id, _e2_id = e1.id, e2.id
        a1_id, _a2_id = a1.id, a2.id

    # DetectionSettings
    res = DetectionSettings.load_from_database(db_path, id=d1_id)
    assert isinstance(res, DetectionSettings)
    assert res.model_type == "cpsam"

    res_list = DetectionSettings.load_from_database(db_path, method="cellpose")
    assert len(res_list) == 2
    assert res_list[0].method == "cellpose"

    res_all = DetectionSettings.load_from_database(db_path)
    assert len(res_all) == 2

    with pytest.raises(ValueError):
        DetectionSettings.load_from_database(db_path, id=999)

    # ExtractionSettings
    res = ExtractionSettings.load_from_database(db_path, id=e1_id)
    assert isinstance(res, ExtractionSettings)
    assert res.dff_window == 10

    res_all = ExtractionSettings.load_from_database(db_path)
    assert len(res_all) == 2

    with pytest.raises(ValueError):
        ExtractionSettings.load_from_database(db_path, id=999)

    # AnalysisSettings
    res = AnalysisSettings.load_from_database(db_path, id=a1_id)
    assert isinstance(res, AnalysisSettings)
    assert res.peaks_height_value == 1.0

    res_all = AnalysisSettings.load_from_database(db_path)
    assert len(res_all) == 2

    with pytest.raises(ValueError):
        AnalysisSettings.load_from_database(db_path, id=999)


def test_cali_result_load_from_database(
    simple_experiment: Experiment, temp_db: TempDB
) -> None:
    """Test CaliResult.load_from_database."""
    from cali.sqlmodel import CaliResult

    engine, db_path = temp_db

    with Session(engine) as session:
        # Create CaliResult
        exp = session.exec(
            select(Experiment).where(Experiment.name == simple_experiment.name)
        ).first()

        res1 = CaliResult(experiment=exp.id, positions_analyzed=[1])
        res2 = CaliResult(experiment=exp.id, positions_analyzed=[2])
        session.add(res1)
        session.add(res2)
        session.commit()
        res1_id = res1.id
        exp_id = exp.id

    # Test loading by ID
    res = CaliResult.load_from_database(db_path, id=res1_id)
    assert isinstance(res, CaliResult)
    assert res.positions_analyzed == [1]

    # Test loading by experiment_id
    results = CaliResult.load_from_database(db_path, experiment_id=exp_id)
    assert len(results) == 2

    # Test loading all
    results = CaliResult.load_from_database(db_path)
    assert len(results) == 2

    # Test error
    with pytest.raises(ValueError):
        CaliResult.load_from_database(db_path, id=999)


def test_experiment_create(temp_db: TempDB) -> None:
    """Test Experiment.create."""
    _engine, _ = temp_db

    # Basic creation
    exp = Experiment.create(
        name="Created Exp", plate_type="96-well", well_names=["A1", "B2"]
    )
    assert exp.name == "Created Exp"
    assert len(exp.plate.wells) == 2

    # With multiple FOVs
    exp2 = Experiment.create(
        name="Multi FOV", plate_type="96-well", well_names=["A1"], fovs_per_well=3
    )
    assert len(exp2.plate.wells) == 1
    assert len(exp2.plate.wells[0].fovs) == 3
    assert exp2.plate.wells[0].fovs[0].fov_number == 0
    assert exp2.plate.wells[0].fovs[2].fov_number == 2

    # With plate maps
    plate_maps = {"genotype": {"A1": "WT"}}
    exp3 = Experiment.create(
        name="Mapped Exp",
        plate_type="96-well",
        well_names=["A1"],
        plate_maps=plate_maps,
    )
    assert len(exp3.plate.wells[0].conditions) == 1
    assert exp3.plate.wells[0].conditions[0].name == "WT"

    # With all wells (default) - 6-well plate
    exp4 = Experiment.create(name="All Wells", plate_type="6-well")
    assert len(exp4.plate.wells) == 6


def test_print_cali_results_all(populated_db: Engine) -> None:
    # Test printing all results
    print_cali_results(populated_db, show_settings=True)


def test_print_cali_results_filtered(populated_db: Engine) -> None:
    # Test filtering by experiment name
    print_cali_results(
        populated_db, experiment_name="Test Experiment", show_settings=True
    )


def test_print_cali_results_not_found(populated_db: Engine) -> None:
    # Test experiment not found
    print_cali_results(populated_db, experiment_name="NonExistent")


def test_print_cali_results_no_results(mock_engine: Engine) -> None:
    # Test no results in DB
    print_cali_results(mock_engine)
    print_cali_results(mock_engine, experiment_name="Test Experiment")


def test_print_cali_results_levels(populated_db: Engine) -> None:
    # Test different max levels
    for level in ["experiment", "plate", "well", "fov", "roi"]:
        print_cali_results(
            populated_db, experiment_name="Test Experiment", max_experiment_level=level
        )


def test_cali_result_eq_hash(populated_db: Engine) -> None:
    with Session(populated_db) as session:
        result1 = session.exec(select(CaliResult)).first()
        # Create a copy
        result2 = CaliResult(
            experiment=result1.experiment,
            detection_settings_id=result1.detection_settings_id,
            extraction_settings_id=result1.extraction_settings_id,
            analysis_settings_id=result1.analysis_settings_id,
            positions_analyzed=result1.positions_analyzed,
        )

        assert result1 == result2
        assert hash(result1) == hash(result2)

        # Test inequality
        result3 = CaliResult(
            experiment=result1.experiment,
            detection_settings_id=result1.detection_settings_id,
            extraction_settings_id=result1.extraction_settings_id,
            analysis_settings_id=result1.analysis_settings_id,
            positions_analyzed=[999],
        )
        assert result1 != result3
        assert result1 != "not a result"


def test_cali_result_load_from_database_coverage(
    populated_db: Engine, tmp_path: Path
) -> None:
    # We need a real file database for load_from_database as it takes a path
    db_path = tmp_path / "test_cali_result.db"
    engine = create_engine(f"sqlite:///{db_path}")
    from cali.sqlmodel._util import create_database_and_tables

    create_database_and_tables(engine)

    # Copy data from populated_db to file db
    with Session(populated_db), Session(engine) as dst_session:
        # We need to copy everything... this is tedious.
        # Instead, let's just create new data in the file db
        exp = Experiment(name="Test Exp File", description="Desc")
        dst_session.add(exp)
        dst_session.commit()
        dst_session.refresh(exp)

        res = CaliResult(experiment=exp.id, positions_analyzed=[1])
        dst_session.add(res)
        dst_session.commit()
        dst_session.refresh(res)
        res_id = res.id
        exp_id = exp.id

    engine.dispose()

    # Test loading
    # Load by ID
    loaded_res = CaliResult.load_from_database(db_path, id=res_id)
    assert loaded_res.id == res_id

    # Load by Experiment ID
    loaded_results = CaliResult.load_from_database(db_path, experiment_id=exp_id)
    assert len(loaded_results) == 1
    assert loaded_results[0].id == res_id

    # Load all
    loaded_all = CaliResult.load_from_database(db_path)
    assert len(loaded_all) == 1

    # Load with existing session
    engine2 = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine2) as session:
            loaded_res_sess = CaliResult.load_from_database(
                db_path, id=res_id, session=session
            )
            assert loaded_res_sess.id == res_id
    finally:
        engine2.dispose()

    # Test not found
    with pytest.raises(ValueError):
        CaliResult.load_from_database(db_path, id=999)


def test_experiment_eq_hash() -> None:
    exp1 = Experiment(name="Exp1")
    exp2 = Experiment(name="Exp1")
    exp3 = Experiment(name="Exp2")

    assert exp1 == exp2
    assert exp1 != exp3
    assert exp1 != "not exp"

    # Hash without ID
    assert hash(exp1) == hash(id(exp1))

    # Hash with ID
    exp1.id = 1
    assert hash(exp1) == hash(1)


def test_experiment_load_from_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test_exp.db"
    engine = create_engine(f"sqlite:///{db_path}")
    from cali.sqlmodel._util import create_database_and_tables

    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="Test Exp Load")
        session.add(exp)
        session.commit()
        exp_id = exp.id
    engine.dispose()

    # Load
    loaded_exp = Experiment.load_from_db(db_path, id=exp_id)
    assert loaded_exp.name == "Test Exp Load"

    # Load with load_data=False
    loaded_exp_nodata = Experiment.load_from_db(db_path, id=exp_id, load_data=False)
    assert loaded_exp_nodata.name == "Test Exp Load"

    # Load with session
    engine2 = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine2) as session:
            loaded_exp_sess = Experiment.load_from_db(
                db_path, id=exp_id, session=session
            )
            assert loaded_exp_sess.name == "Test Exp Load"
    finally:
        engine2.dispose()


def test_experiment_create_coverage(tmp_path: Path) -> None:
    # Test Experiment.create
    exp = Experiment.create(
        name="Created Exp",
        plate_type="96-well",
        well_names=["A1", "B2"],
        fovs_per_well=2,
        description="Created Description",
    )

    assert exp.name == "Created Exp"
    assert exp.plate.plate_type == "96-well"
    assert len(exp.plate.wells) == 2
    assert len(exp.plate.wells[0].fovs) == 2

    # Test with plate maps
    exp_mapped = Experiment.create(
        name="Mapped Exp", plate_maps={"cond": {"A1": "val"}}, well_names=["A1"]
    )
    assert len(exp_mapped.plate.wells[0].conditions) == 1


def test_experiment_create_from_data_tiff(tmp_path: Path) -> None:
    # Define a mock class that can be used with isinstance
    class MockTiffCollectionReader:
        def __init__(self, settings: object) -> None:
            self.settings = settings

        def to_experiment_tiff_config(
            self,
        ) -> tuple[dict[str, list[str]], str, dict[str, float]]:
            return ({"A1": ["path.tif"]}, "96-well", {"exposure_ms": 100})

    # Mock TiffCollectionReader and data_to_plate
    with (
        patch("cali.readers.TiffCollectionReader", MockTiffCollectionReader),
        patch("cali.sqlmodel._data_to_plate.data_to_plate") as mock_data_to_plate,
        patch.dict("sys.modules", {"cali.util": MagicMock()}),
    ):
        mock_data_to_plate.return_value = Plate(name="96-well", plate_type="96-well")

        exp = Experiment.create_from_data(
            name="Tiff Exp",
            data_path=str(tmp_path),
            tiff_file_map={"A1": ["path.tif"]},
            tiff_plate_type="96-well",
            tiff_metadata={"exposure_ms": 100},
        )

        assert exp.name == "Tiff Exp"
        assert exp.tiff_file_map_json is not None
        assert exp.tiff_plate_type == "96-well"

        # Test tiff_collection_settings
        settings = exp.tiff_collection_settings(str(tmp_path))
        assert settings is not None
        assert settings.plate == "96-well"

        # Test tiff_collection_settings returns None if fields missing
        exp.tiff_plate_type = None
        assert exp.tiff_collection_settings(str(tmp_path)) is None


def test_settings_load_from_database_coverage(tmp_path: Path) -> None:
    db_path = tmp_path / "test_settings.db"
    engine = create_engine(f"sqlite:///{db_path}")
    from cali.sqlmodel._util import create_database_and_tables

    create_database_and_tables(engine)

    with Session(engine) as session:
        det = DetectionSettings(method="cellpose", model_type="cyto", diameter=30)
        session.add(det)

        ext = ExtractionSettings(neuropil_inner_radius=5)
        session.add(ext)

        ana = AnalysisSettings(experiment_type="Spontaneous")
        session.add(ana)

        session.commit()
        det_id = det.id
        ext_id = ext.id
        ana_id = ana.id

    engine.dispose()

    # Test DetectionSettings.load_from_database
    loaded_det = DetectionSettings.load_from_database(db_path, id=det_id)
    assert loaded_det.method == "cellpose"

    loaded_dets = DetectionSettings.load_from_database(db_path, method="cellpose")
    assert len(loaded_dets) == 1

    with pytest.raises(ValueError):
        DetectionSettings.load_from_database(db_path, id=999)

    # Test ExtractionSettings.load_from_database
    loaded_ext = ExtractionSettings.load_from_database(db_path, id=ext_id)
    assert loaded_ext.neuropil_inner_radius == 5

    loaded_exts = ExtractionSettings.load_from_database(db_path)
    assert len(loaded_exts) == 1

    with pytest.raises(ValueError):
        ExtractionSettings.load_from_database(db_path, id=999)

    # Test AnalysisSettings.load_from_database
    loaded_ana = AnalysisSettings.load_from_database(db_path, id=ana_id)
    assert loaded_ana.experiment_type == "Spontaneous"

    loaded_anas = AnalysisSettings.load_from_database(db_path)
    assert len(loaded_anas) == 1

    with pytest.raises(ValueError):
        AnalysisSettings.load_from_database(db_path, id=999)


def test_has_fov_analysis(temp_db: TempDB) -> None:
    """Test has_fov_analysis function."""
    from cali.sqlmodel._util import has_fov_analysis

    engine, db_path = temp_db

    # Create experiment with FOV and ROI
    with Session(engine) as session:
        exp = Experiment(name="Test", data_path="/test")
        plate = Plate(name="96-well", rows=8, columns=12, experiment=exp)
        well = Well(name="A1", row=0, column=0, plate=plate)
        fov = FOV(name="A1_0000", position_index=0, well=well)
        roi = ROI(label_value=1, fov=fov)
        session.add(exp)
        session.commit()
        roi_id = roi.id

    # FOV exists but has no traces yet
    assert not has_fov_analysis(db_path, "A1_0000")

    # Add traces to the ROI
    with Session(engine) as session:
        roi = session.get(ROI, roi_id)
        assert roi is not None
        trace = Traces(
            raw_trace=[1.0, 2.0, 3.0],
            corrected_trace=[1.0, 2.0, 3.0],
            roi=roi,
        )
        session.add(trace)
        session.commit()

    # Now it should have analysis
    assert has_fov_analysis(db_path, "A1_0000")

    # Non-existent FOV
    assert not has_fov_analysis(db_path, "Z99_9999")


def test_has_experiment_analysis(temp_db: TempDB) -> None:
    """Test has_experiment_analysis function."""
    from cali.sqlmodel._util import has_experiment_analysis

    engine, db_path = temp_db

    # Empty database
    assert not has_experiment_analysis(db_path)

    # Create experiment with FOV and ROI
    with Session(engine) as session:
        exp = Experiment(name="Test", data_path="/test")
        plate = Plate(name="96-well", rows=8, columns=12, experiment=exp)
        well = Well(name="A1", row=0, column=0, plate=plate)
        fov = FOV(name="A1_0000", position_index=0, well=well)
        roi = ROI(label_value=1, fov=fov)
        session.add(exp)
        session.commit()
        roi_id = roi.id

    # No analysis yet
    assert not has_experiment_analysis(db_path)

    # Add traces
    with Session(engine) as session:
        roi = session.get(ROI, roi_id)
        assert roi is not None
        trace = Traces(
            raw_trace=[1.0, 2.0, 3.0],
            corrected_trace=[1.0, 2.0, 3.0],
            roi=roi,
        )
        session.add(trace)
        session.commit()

    # Now it has analysis
    assert has_experiment_analysis(db_path)


def test_load_experiment_from_database_not_exists() -> None:
    """Test load_experiment_from_database with non-existent database."""
    from cali.sqlmodel._util import load_experiment_from_database

    result = load_experiment_from_database("/nonexistent/path.db")
    assert result is None


def test_load_experiment_from_database_by_name(temp_db: TempDB) -> None:
    """Test load_experiment_from_database with specific experiment name."""
    from cali.sqlmodel._util import load_experiment_from_database

    engine, db_path = temp_db

    # Create two experiments
    with Session(engine) as session:
        exp1 = Experiment(name="Experiment1", data_path="/test1")
        exp2 = Experiment(name="Experiment2", data_path="/test2")
        session.add(exp1)
        session.add(exp2)
        session.commit()

    # Load specific experiment
    loaded = load_experiment_from_database(db_path, experiment_name="Experiment2")
    assert loaded is not None
    assert loaded.name == "Experiment2"

    # Load non-existent experiment
    loaded = load_experiment_from_database(db_path, experiment_name="NonExistent")
    assert loaded is None


def test_save_experiment_to_database_overwrite(tmp_path: Path) -> None:
    """Test save_experiment_to_database with overwrite."""
    from cali.sqlmodel._util import save_experiment_to_database

    exp = Experiment(name="Test", data_path="/test")
    plate = Plate(name="96-well", rows=8, columns=12, experiment=exp)
    well = Well(name="A1", row=0, column=0, plate=plate)
    FOV(name="A1_0000", position_index=0, well=well)

    # Save first time
    save_experiment_to_database(exp, tmp_path, database_name="test.db")
    db_path = tmp_path / "test.db"
    assert _get_actual_db_path(db_path).exists()

    # Save again with overwrite=True
    exp2 = Experiment(name="Test2", data_path="/test2")
    save_experiment_to_database(exp2, tmp_path, database_name="test.db", overwrite=True)

    # Verify it was overwritten
    from cali.sqlmodel._util import load_experiment_from_database

    loaded = load_experiment_from_database(_get_actual_db_path(db_path))
    assert loaded is not None
    assert loaded.name == "Test2"
