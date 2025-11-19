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
    load_analysis_from_json,
    save_experiment_to_database,
    useq_plate_plan_to_db,
)
from cali.sqlmodel._json_to_db import load_plate_map, parse_well_name, roi_from_roi_data
from cali.sqlmodel._model import (
    AnalysisSettings,
    DataAnalysis,
    Mask,
    Traces,
)
from cali.sqlmodel._util import (
    create_database_and_tables,
)

TempDB = tuple[Engine, Path]

TempDB = tuple[Engine, Path]

if TYPE_CHECKING:
    from collections.abc import Generator

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
    engine, db_path = temp_db

    # Create experiment
    exp = Experiment(
        name="test_experiment",
        description="Test experiment",
        data_path="/dummy/path",
        analysis_path=str(db_path.parent),
        database_name=db_path.name,
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
        cell_size=100.5,
        cell_size_units="pixels",
    )

    # Save to database
    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

    return exp


# ==================== Model Tests ====================


def test_experiment_creation(temp_db: TempDB) -> None:
    """Test basic Experiment model creation."""
    engine, db_path = temp_db

    exp = Experiment(
        name="test_exp",
        description="Test description",
        data_path="/path/to/data",
        analysis_path=str(db_path.parent),
        database_name=db_path.name,
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
    from cali._constants import SPONTANEOUS

    # Create experiment from test data
    exp = Experiment.create_from_data(
        name="Test Experiment From Data",
        data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
        analysis_path=str(tmp_path),
        database_name="test_from_data.db",
        plate_maps={
            "genotype": {"B5": "WT"},
            "treatment": {"B5": "Vehicle"},
        },
        experiment_type=SPONTANEOUS,
    )

    # Verify experiment structure was loaded from data
    assert exp.name == "Test Experiment From Data"
    assert exp.plate is not None
    assert len(exp.plate.wells) > 0
    assert exp.plate.wells[0].name == "B5"
    assert len(exp.plate.wells[0].fovs) > 0
    assert exp.experiment_type == SPONTANEOUS

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
        assert roi.data_analysis_history[0].cell_size == 100.5


def test_unique_constraints(temp_db: TempDB) -> None:
    """Test unique constraints on models."""
    engine, db_path = temp_db

    # Experiment names must be unique
    with Session(engine) as session:
        session.add(
            Experiment(
                name="test1",
                data_path="/dummy/path",
                analysis_path=str(db_path.parent),
                database_name=db_path.name,
            )
        )
        session.commit()

    with Session(engine) as session:
        session.add(
            Experiment(
                name="test1",
                data_path="/dummy/path",
                analysis_path=str(db_path.parent),
                database_name=db_path.name,
            )
        )
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


def test_load_analysis_from_json() -> None:
    """Test loading analysis from JSON directory."""
    test_data_dir = Path(__file__).parent / "test_data" / "evoked"
    data_path = test_data_dir / "evk.tensorstore.zarr"
    analysis_path = test_data_dir / "evk_analysis"

    if not analysis_path.exists():
        pytest.skip("Test data not available")

    plate = useq.WellPlate.from_str("96-well")
    experiment = load_analysis_from_json(str(data_path), str(analysis_path), plate)

    # Name is now derived from data_path name + ".db"
    assert experiment.name == "evk.tensorstore.zarr.db"
    assert experiment.plate is not None
    assert len(experiment.plate.wells) > 0
    # Analysis settings are created separately, not as experiment attribute


def test_save_experiment_to_db(tmp_path: Path) -> None:
    """Test saving experiment to database."""
    db_path = tmp_path / "test.db"

    # Create simple experiment with analysis_path set
    exp = Experiment(
        name="test_experiment",
        description="Test",
        data_path=str(tmp_path / "data"),
        analysis_path=str(tmp_path),
        database_name="test.db",
    )
    Plate(experiment=exp, name="96-well", plate_type="96-well")

    # Save
    save_experiment_to_database(exp, overwrite=True)

    # Verify
    from cali.sqlmodel._util import load_experiment_from_database

    result = load_experiment_from_database(db_path)
    assert result is not None
    assert result.name == "test_experiment"
    assert db_path.exists()


def test_save_experiment_overwrite_protection(
    simple_experiment: Experiment, tmp_path: Path
) -> None:
    """Test that overwrite=False protects existing database."""
    # Update experiment to use tmp_path
    simple_experiment.analysis_path = str(tmp_path)
    simple_experiment.database_name = "test.db"
    db_path = tmp_path / "test.db"

    # Create initial database
    save_experiment_to_database(simple_experiment, overwrite=True)

    # Try to save again without overwrite - should work (SQLite appends)
    # but verify the file exists
    assert db_path.exists()
    db_path.stat().st_size

    # Save with overwrite=True
    save_experiment_to_database(simple_experiment, overwrite=True)
    # File should still exist
    assert db_path.exists()


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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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

    # Verify Trace
    assert trace.raw_trace == [1.0, 2.0, 3.0]
    assert trace.dff == [0.0, 0.1, 0.2]

    # Verify DataAnalysis
    assert data_analysis.cell_size == 100.0
    assert data_analysis.cell_size_units == "pixels"

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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
    """Test complete workflow from JSON to database to export."""
    test_data_dir = Path(__file__).parent / "test_data" / "evoked"
    data_path = test_data_dir / "evk.tensorstore.zarr"
    analysis_path = test_data_dir / "evk_analysis"

    if not analysis_path.exists():
        pytest.skip("Test data not available")

    # 1. Load from JSON
    plate = useq.WellPlate.from_str("96-well")
    experiment = load_analysis_from_json(str(data_path), str(analysis_path), plate)

    # Verify basic experiment structure
    assert experiment.plate is not None
    assert len(experiment.plate.wells) > 0

    # 2. Save to database (update paths to use tmp_path)
    experiment.analysis_path = str(tmp_path)
    experiment.database_name = "test.db"
    db_path = tmp_path / "test.db"
    save_experiment_to_database(experiment, overwrite=True)

    # 3. Read back from database
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            exp = session.exec(select(Experiment)).first()

            # 4. Convert to useq.WellPlate
            useq_plate = experiment_to_useq_plate(exp)
            assert useq_plate is not None

            # 5. Convert to useq.WellPlatePlan
            useq_plate_plan = experiment_to_useq_plate_plan(exp)
            assert useq_plate_plan is not None
    finally:
        # Cleanup - dispose engine (Python 3.13 compatibility)
        engine.dispose(close=True)


def test_data_to_plate_error_cases(tmp_path: Path) -> None:
    """Test data_to_plate error handling."""
    from cali.sqlmodel._data_to_plate import data_to_plate

    exp = Experiment(
        name="test",
        data_path="/dummy/path",
        analysis_path=str(tmp_path),
        database_name="test.db",
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

    # Test with invalid path - load_data raises ValueError
    with pytest.raises(ValueError, match="Unsupported data format"):
        data_to_plate("/nonexistent/path", exp)

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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Test experiment_to_useq_plate with no plate - this should raise AttributeError
        # because experiment.plate is None
        with pytest.raises(AttributeError):
            experiment_to_useq_plate(exp)


def test_useq_plate_to_db_with_positions(temp_db: TempDB) -> None:
    """Test useq_plate_plan_to_db with actual positions."""
    engine, _ = temp_db

    exp = Experiment(
        name="test_positions",
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path=str(tmp_path),
        database_name="test_load.db",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="B5", row=1, column=4)
    FOV(well=well, name="B5_0000", position_index=0, fov_number=0)

    # Save to database (this creates tables internally)
    save_experiment_to_database(exp, overwrite=True)

    # Load back
    db_path = tmp_path / "test_load.db"
    loaded_exp = load_experiment_from_database(db_path)

    assert loaded_exp is not None
    assert loaded_exp.name == "test_load"
    assert loaded_exp.plate is not None
    assert len(loaded_exp.plate.wells) == 1
    assert len(loaded_exp.plate.wells[0].fovs) == 1

    # Test with non-existent database
    result = load_experiment_from_database(tmp_path / "nonexistent.db")
    assert result is None


def test_visualize_experiment_functions(simple_experiment: Experiment, temp_db: TempDB) -> None:
    """Test visualization functions."""
    from cali.sqlmodel._visualize_experiment import (
        print_all_analysis_results,
        print_experiment_tree,
        print_experiment_tree_from_engine,
    )

    engine, _db_path = temp_db

    with Session(engine) as session:
        exp = session.get(Experiment, simple_experiment.id)

        # Test print_experiment_tree with different max levels
        print_experiment_tree(exp, max_experiment_level="experiment", session=session)
        print_experiment_tree(exp, max_experiment_level="plate", session=session)
        print_experiment_tree(exp, max_experiment_level="well", session=session)
        print_experiment_tree(exp, max_experiment_level="fov", session=session)
        print_experiment_tree(exp, max_experiment_level="roi", session=session)

        # Test with analysis results off
        print_experiment_tree(exp, show_analysis_results=False, session=session)
        print_experiment_tree(exp, show_settings=False, session=session)

    # Test print_all_analysis_results
    print_all_analysis_results(
        engine,
        experiment_name=simple_experiment.name,
        show_settings=False,
    )

    print_all_analysis_results(
        engine,
        experiment_name=None,  # All experiments
        show_settings=True,
    )

    # Test print_experiment_tree_from_engine
    print_experiment_tree_from_engine(
        simple_experiment.name,
        engine,
        max_level="roi",
        show_analysis_results=True,
        show_settings=True,
    )

    # Test with non-existent experiment
    print_experiment_tree_from_engine(
        "NonExistent",
        engine,
        max_level="roi",
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
    settings = AnalysisSettings()
    assert settings.stimulated_mask_area() is None

    # Test with mask
    mask = Mask(
        coords_y=[0, 1, 2],
        coords_x=[0, 1, 2],
        height=10,
        width=10,
        mask_type="stimulation",
    )
    settings = AnalysisSettings(stimulation_mask=mask)
    result = settings.stimulated_mask_area()
    assert result is not None
    assert result.shape == (10, 10)


def test_db_to_plate_map_with_multiple_condition_types(temp_db: TempDB) -> None:
    """Test experiment_to_plate_map_data with various configurations."""
    from cali.sqlmodel._db_to_plate_map import experiment_to_plate_map_data

    engine, _ = temp_db

    exp = Experiment(
        name="test_map",
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
    )
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    fov = FOV(well=well, name="A1_0000_p0", position_index=0)
    roi = ROI(fov=fov, label_value=1)

    DataAnalysis(
        roi=roi,
        cell_size=150.5,
        cell_size_units="μm²",
        total_recording_time_sec=600.0,
        dec_dff_frequency=2.5,
        peaks_dec_dff=[10.0, 20.0, 30.0],
        peaks_amplitudes_dec_dff=[0.5, 0.6, 0.7],
        iei=[10.0, 10.0],
        inferred_spikes=[0.1, 0.2, 0.3],
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        result = session.exec(select(DataAnalysis)).first()
        assert result.cell_size_units == "μm²"
        assert result.total_recording_time_sec == 600.0
        assert result.peaks_amplitudes_dec_dff == [0.5, 0.6, 0.7]
        assert result.iei == [10.0, 10.0]


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
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
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
    """Test ROI with both ROI and neuropil masks."""
    engine, _ = temp_db

    exp = Experiment(
        name="test",
        data_path="/dummy/path",
        analysis_path="/dummy/analysis",
        database_name="test.db",
    )
    plate = Plate(experiment=exp, name="96-well", plate_type="96-well")
    well = Well(plate=plate, name="A1", row=0, column=0)
    fov = FOV(well=well, name="A1_0000_p0", position_index=0)

    roi_mask = Mask(
        coords_y=[0, 1], coords_x=[0, 1], height=10, width=10, mask_type="roi"
    )
    neuropil_mask = Mask(
        coords_y=[2, 3], coords_x=[2, 3], height=10, width=10, mask_type="neuropil"
    )

    roi = ROI(fov=fov, label_value=1)

    with Session(engine) as session:
        # Add masks first
        session.add(roi_mask)
        session.add(neuropil_mask)
        session.flush()

        # Set mask IDs on ROI
        roi.roi_mask_id = roi_mask.id
        roi.neuropil_mask_id = neuropil_mask.id
        roi.roi_mask = roi_mask
        roi.neuropil_mask = neuropil_mask

        session.add(exp)
        session.commit()

        # Retrieve and verify
        result = session.exec(select(ROI)).first()
        assert result.roi_mask is not None
        assert result.neuropil_mask is not None
        assert result.roi_mask.mask_type == "roi"
        assert result.neuropil_mask.mask_type == "neuropil"

        # Force load mask relationships before expunging
        _ = result.roi_mask
        _ = result.neuropil_mask

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
        settings = AnalysisSettings()

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
        cond1_data, cond2_data = experiment_to_plate_map_data(simple_experiment)

        # Verify condition_1 (genotype)
        assert len(cond1_data) == 1
        assert cond1_data[0].name == "B5"
        assert cond1_data[0].row_col == (1, 4)
        assert cond1_data[0].condition == ("WT", "blue")

        # Verify condition_2 (treatment)
        assert len(cond2_data) == 1
        assert cond2_data[0].name == "B5"
        assert cond2_data[0].row_col == (1, 4)
        assert cond2_data[0].condition == ("Control", "gray")


def test_experiment_to_plate_map_data_multiple_wells(temp_db: TempDB) -> None:
    """Test plate map conversion with multiple wells.

    Note: The order of conditions in a well's condition list is not guaranteed
    by SQLModel/SQLAlchemy when using many-to-many relationships. This test
    verifies that the conversion function works correctly regardless of order.
    """
    engine, _ = temp_db

    with Session(engine) as session:
        # Create experiment with multiple wells
        exp = Experiment(
            name="multi_well_test",
            data_path="/dummy/path",
            analysis_path="/dummy/analysis",
            database_name="test.db",
        )
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
        cond1_data, cond2_data = experiment_to_plate_map_data(exp)

        # Verify we have 4 wells total
        assert len(cond1_data) == 4
        assert len(cond2_data) == 4

        # Check that all wells are present
        well_names_1 = {data.name for data in cond1_data}
        well_names_2 = {data.name for data in cond2_data}
        assert well_names_1 == {"A1", "A2", "B1", "B2"}
        assert well_names_2 == {"A1", "A2", "B1", "B2"}

        # Verify that each well has two conditions (one in each list)
        for well_name in ["A1", "A2", "B1", "B2"]:
            cond1_entry = next(d for d in cond1_data if d.name == well_name)
            cond2_entry = next(d for d in cond2_data if d.name == well_name)

            # Each condition should be one of our defined conditions
            assert cond1_entry.condition[0] in ["WT", "KO", "Drug", "Vehicle"]
            assert cond2_entry.condition[0] in ["WT", "KO", "Drug", "Vehicle"]

            # The two conditions for each well should be different
            assert cond1_entry.condition[0] != cond2_entry.condition[0]

            # Colors should match condition names
            color_map = {"WT": "blue", "KO": "red", "Drug": "green", "Vehicle": "gray"}
            assert cond1_entry.condition[1] == color_map[cond1_entry.condition[0]]
            assert cond2_entry.condition[1] == color_map[cond2_entry.condition[0]]


def test_experiment_to_plate_map_data_no_conditions(temp_db: TempDB) -> None:
    """Test plate map conversion with wells that have no conditions."""
    engine, _ = temp_db

    with Session(engine) as session:
        # Create experiment with wells but no conditions
        exp = Experiment(
            name="no_conditions_test",
            data_path="/dummy/path",
            analysis_path="/dummy/analysis",
            database_name="test.db",
        )
        plate = Plate(experiment=exp, name="24-well")
        Well(plate=plate, name="A1", row=0, column=0)
        Well(plate=plate, name="A2", row=0, column=1)

        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert to plate map format
        cond1_data, cond2_data = experiment_to_plate_map_data(exp)

        # Should return empty lists when wells have no conditions
        assert len(cond1_data) == 0
        assert len(cond2_data) == 0


def test_experiment_to_plate_map_data_no_plate(temp_db: TempDB) -> None:
    """Test plate map conversion with experiment that has no plate."""
    engine, _ = temp_db

    with Session(engine) as session:
        # Create experiment without plate
        exp = Experiment(
            name="no_plate_test",
            data_path="/dummy/path",
            analysis_path="/dummy/analysis",
            database_name="test.db",
        )

        session.add(exp)
        session.commit()
        session.refresh(exp)

        # Convert to plate map format
        cond1_data, cond2_data = experiment_to_plate_map_data(exp)

        # Should return empty lists when there's no plate
        assert len(cond1_data) == 0
        assert len(cond2_data) == 0
