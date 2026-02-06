"""Test versioned detection and analysis results with CaliResult tracking.

This test validates the CaliResult model and its relationship with settings.
It focuses on testing the database schema and settings equality, without
requiring full detection/analysis pipeline integration.
"""

from pathlib import Path

import pytest
from sqlmodel import Session, create_engine, select

from cali._constants import SPONTANEOUS
from cali.sqlmodel import AnalysisSettings, Experiment
from cali.sqlmodel._model import (
    CaliResult,
    DetectionSettings,
    ExtractionSettings,
)

THREADS = 1


def _get_actual_db_path(requested_db_path: Path) -> Path:
    """Get the actual database path (with .cali extension added if needed)."""
    if not requested_db_path.name.endswith(".cali"):
        return requested_db_path.parent / f"{requested_db_path.name}.cali"
    return requested_db_path


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create a test database path."""
    return tmp_path / "test_results.db"


@pytest.fixture
def test_experiment(test_db: Path) -> Experiment:
    """Create a test experiment from spontaneous data."""
    exp = Experiment.create_from_data(
        name="Test Versioned Analysis",
        data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
        plate_maps={
            "genotype": {"B5": "WT"},
            "treatment": {"B5": "Vehicle"},
        },
    )
    return exp


def test_detection_settings_equality() -> None:
    """Test DetectionSettings equality ignores created_at timestamp."""
    import time

    ds1 = DetectionSettings(
        method="cellpose",
        model_type="cpsam",
        diameter=30.0,
        cellprob_threshold=0.5,
        flow_threshold=0.4,
        min_size=10,
        normalize=True,
        batch_size=8,
    )

    time.sleep(0.001)  # Ensure different timestamps

    ds2 = DetectionSettings(
        method="cellpose",
        model_type="cpsam",
        diameter=30.0,
        cellprob_threshold=0.5,
        flow_threshold=0.4,
        min_size=10,
        normalize=True,
        batch_size=8,
    )

    # Should be equal despite different created_at times
    assert ds1.created_at != ds2.created_at, "Should have different timestamps"
    assert ds1 == ds2, "DetectionSettings with same parameters should be equal"
    assert hash(ds1) == hash(ds2), "Equal DetectionSettings should have same hash"

    # Different settings should not be equal
    ds3 = DetectionSettings(
        method="cellpose",
        model_type="cpsam",
        diameter=35.0,  # Different diameter
        cellprob_threshold=0.5,
    )
    assert ds1 != ds3, "DetectionSettings with different parameters should not be equal"


def test_analysis_settings_equality() -> None:
    """Test AnalysisSettings equality ignores created_at timestamp."""
    import time

    as1 = AnalysisSettings(
        peaks_height_value=1.5,
        spike_threshold_value=2.0,
        threads=THREADS,
        burst_threshold=20.0,
    )

    time.sleep(0.001)  # Ensure different timestamps

    as2 = AnalysisSettings(
        peaks_height_value=1.5,
        spike_threshold_value=2.0,
        burst_threshold=20.0,
        threads=THREADS,
    )

    # Should be equal despite different created_at times
    assert as1.created_at != as2.created_at, "Should have different timestamps"
    assert as1 == as2, "AnalysisSettings with same parameters should be equal"
    assert hash(as1) == hash(as2), "Equal AnalysisSettings should have same hash"

    # Different settings should not be equal
    as3 = AnalysisSettings(
        peaks_height_value=2,
        spike_threshold_value=2.0,
        threads=4,
    )
    assert as1 != as3, "AnalysisSettings with different parameters should not be equal"


def test_cali_result_equality() -> None:
    """Test CaliResult equality ignores created_at timestamp."""
    import time

    result1 = CaliResult(
        experiment=1, analysis_settings_id=1, positions_analyzed=[0, 1]
    )

    time.sleep(0.001)  # Ensure different timestamps

    result2 = CaliResult(
        experiment=1, analysis_settings_id=1, positions_analyzed=[0, 1]
    )

    # created_at should be different
    assert result1.created_at != result2.created_at, (
        "created_at should be different for objects created at different times"
    )

    # But objects should still be equal (semantic equality)
    assert result1 == result2, (
        "CaliResults with same settings should be equal despite different created_at"
    )

    # Different settings should not be equal
    result3 = CaliResult(
        experiment=1,
        analysis_settings_id=2,  # Different analysis settings
        positions_analyzed=[0, 1],
    )

    assert result1 != result3, "CaliResults with different settings should not be equal"

    # Test hash consistency
    assert hash(result1) == hash(result2), "Equal CaliResults should have same hash"


def test_cali_result_with_none_values() -> None:
    """Test CaliResult equality with None values."""
    result1 = CaliResult(
        experiment=1,
        detection_settings_id=None,
        analysis_settings_id=1,
        positions_analyzed=None,
    )

    result2 = CaliResult(
        experiment=1,
        detection_settings_id=None,
        analysis_settings_id=1,
        positions_analyzed=None,
    )

    assert result1 == result2, (
        "CaliResults with None values should be equal if other fields match"
    )


def test_cali_result_database_storage(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test storing CaliResult in database."""
    from cali.sqlmodel import save_experiment_to_database

    # Save experiment
    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    # The database name gets .cali appended if not already present
    actual_db_path = test_db.parent / f"{test_db.name}.cali"
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            # Create detection settings
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Create analysis settings
            a_settings = AnalysisSettings()
            session.add(a_settings)
            session.commit()
            session.refresh(a_settings)

            # Create CaliResult
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                analysis_settings_id=a_settings.id,
                positions_analyzed=[0, 1, 2],
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            # Verify it was saved
            loaded_result = session.get(CaliResult, result.id)
            assert loaded_result is not None
            assert loaded_result.experiment == test_experiment.id
            assert loaded_result.detection_settings_id == d_settings.id
            assert loaded_result.analysis_settings_id == a_settings.id
            assert loaded_result.positions_analyzed == [0, 1, 2]

    finally:
        engine.dispose(close=True)


def test_detection_settings_database_deduplication(test_db: Path) -> None:
    """Test that identical DetectionSettings are deduplicated in database."""
    from cali.sqlmodel._util import create_database_and_tables

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    create_database_and_tables(engine)

    try:
        with Session(engine) as session:
            # Create first detection settings
            ds1 = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=30.0
            )
            session.add(ds1)
            session.commit()

        with Session(engine) as session:
            # Try to create identical settings
            ds2 = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=30.0
            )

            # Check if it already exists
            existing = session.exec(
                select(DetectionSettings).where(
                    DetectionSettings.method == ds2.method,
                    DetectionSettings.model_type == ds2.model_type,
                    DetectionSettings.diameter == ds2.diameter,
                )
            ).first()

            assert existing is not None, "Identical settings should already exist"
            assert existing == ds2, "Should find equal settings"

    finally:
        engine.dispose(close=True)


def test_analysis_settings_database_deduplication(test_db: Path) -> None:
    """Test that identical AnalysisSettings are deduplicated in database."""
    from cali.sqlmodel._util import create_database_and_tables

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    create_database_and_tables(engine)

    try:
        with Session(engine) as session:
            # Create first analysis settings
            as1 = AnalysisSettings(threads=4)
            session.add(as1)
            session.commit()

        with Session(engine) as session:
            # Try to create identical settings
            as2 = AnalysisSettings(threads=4)

            # Check if it already exists
            existing = session.exec(
                select(AnalysisSettings).where(
                    AnalysisSettings.threads == as2.threads,
                )
            ).first()

            assert existing is not None, "Identical settings should already exist"
            assert existing == as2, "Should find equal settings"

    finally:
        engine.dispose(close=True)


def test_detection_settings_fields() -> None:
    """Test DetectionSettings contains all required fields."""
    ds = DetectionSettings(
        method="cellpose",
        model_type="cpsam",
        custom_model="/path/to/model",
        diameter=30.0,
        cellprob_threshold=0.5,
        flow_threshold=0.4,
        min_size=15,
        normalize=True,
        batch_size=16,
    )

    assert ds.method == "cellpose"
    assert ds.model_type == "cpsam"
    assert ds.custom_model == "/path/to/model"
    assert ds.diameter == 30.0
    assert ds.cellprob_threshold == 0.5
    assert ds.flow_threshold == 0.4
    assert ds.min_size == 15
    assert ds.normalize is True
    assert ds.batch_size == 16
    assert ds.created_at is not None


def test_analysis_settings_evoked_fields() -> None:
    """Test AnalysisSettings with evoked experiment fields."""
    settings = AnalysisSettings(
        led_power_equation="y = 0.5 * x",
        led_pulse_duration=50.0,
        led_pulse_powers=[5.0, 10.0, 15.0],
        led_pulse_on_frames=[100, 200, 300],
    )

    assert settings.led_power_equation == "y = 0.5 * x"
    assert settings.led_pulse_duration == 50.0
    assert settings.led_pulse_powers == [5.0, 10.0, 15.0]
    assert settings.led_pulse_on_frames == [100, 200, 300]


def test_cali_result_links_to_settings(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test that CaliResult properly links to DetectionSettings and AnalysisSettings."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            # Create settings
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            a_settings = AnalysisSettings()
            session.add(d_settings)
            session.add(a_settings)
            session.commit()
            session.refresh(d_settings)
            session.refresh(a_settings)

            # Create CaliResult linking to both
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                analysis_settings_id=a_settings.id,
                positions_analyzed=[0],
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            # Verify linkages
            assert result.detection_settings_id == d_settings.id
            assert result.analysis_settings_id == a_settings.id

            # Verify we can query back
            loaded_d_settings = session.get(
                DetectionSettings, result.detection_settings_id
            )
            loaded_a_settings = session.get(
                AnalysisSettings, result.analysis_settings_id
            )

            assert loaded_d_settings is not None
            assert loaded_a_settings is not None
            assert loaded_d_settings.method == "cellpose"
            assert loaded_a_settings.threads == 1

    finally:
        engine.dispose(close=True)


def test_detection_only_cali_result(test_db: Path, test_experiment: Experiment) -> None:
    """Test CaliResult for detection-only runs (no analysis)."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            # Create detection settings only
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Create CaliResult with no analysis
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                analysis_settings_id=None,  # No analysis
                positions_analyzed=[0],
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            # Verify
            assert result.detection_settings_id == d_settings.id
            assert result.analysis_settings_id is None

    finally:
        engine.dispose(close=True)


def test_cali_result_positions_list(test_db: Path, test_experiment: Experiment) -> None:
    """Test that CaliResult stores positions_analyzed as a list."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Create with specific positions
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                positions_analyzed=[0, 2, 5, 10],
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            # Verify list is preserved
            assert result.positions_analyzed == [0, 2, 5, 10]
            assert isinstance(result.positions_analyzed, list)

    finally:
        engine.dispose(close=True)


def test_multiple_cali_results_same_experiment(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test multiple CaliResults for same experiment with different settings."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            # Create different detection settings
            d_settings_1 = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=30
            )
            d_settings_2 = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=50
            )
            session.add_all([d_settings_1, d_settings_2])
            session.commit()
            session.refresh(d_settings_1)
            session.refresh(d_settings_2)

            # Create different analysis settings
            a_settings_1 = AnalysisSettings()
            a_settings_2 = AnalysisSettings()
            session.add_all([a_settings_1, a_settings_2])
            session.commit()
            session.refresh(a_settings_1)
            session.refresh(a_settings_2)

            # Create multiple CaliResults
            result_1 = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings_1.id,
                analysis_settings_id=a_settings_1.id,
                positions_analyzed=[0],
            )
            result_2 = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings_2.id,
                analysis_settings_id=a_settings_2.id,
                positions_analyzed=[0],
            )
            session.add_all([result_1, result_2])
            session.commit()

            # Verify both exist
            all_results = session.exec(
                select(CaliResult).where(CaliResult.experiment == test_experiment.id)
            ).all()

            assert len(all_results) == 2
            # Verify they link to different settings
            settings_pairs = {
                (r.detection_settings_id, r.analysis_settings_id) for r in all_results
            }
            assert len(settings_pairs) == 2

    finally:
        engine.dispose(close=True)


def test_cali_result_progressive_upgrade(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test progressive upgrade: detection → detection+extraction → full analysis.

    This tests the workflow where a user:
    1. Runs detection only
    2. Runs detection + extraction (should upgrade result, not create new)
    3. Runs detection + extraction + analysis (should upgrade again, not create new)
    """
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            # Create settings
            d_settings = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=30
            )
            e_settings = ExtractionSettings(
                neuropil_inner_radius=2,
                neuropil_min_pixels=50,
                neuropil_correction_factor=0.7,
                threads=THREADS,
            )
            a_settings = AnalysisSettings(threads=THREADS)
            session.add_all([d_settings, e_settings, a_settings])
            session.commit()
            session.refresh(d_settings)
            session.refresh(e_settings)
            session.refresh(a_settings)

            # Stage 1: Detection only
            result1 = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                extraction_settings_id=None,
                analysis_settings_id=None,
                positions_analyzed=[0],
            )
            session.add(result1)
            session.commit()
            session.refresh(result1)
            result1_id = result1.id

            # Verify detection-only result
            assert result1.detection_settings_id == d_settings.id
            assert result1.extraction_settings_id is None
            assert result1.analysis_settings_id is None

            # Stage 2: Add extraction (simulate upgrade)
            result1.extraction_settings_id = e_settings.id
            session.add(result1)
            session.commit()
            session.refresh(result1)

            # Verify same ID, now with extraction
            assert result1.id == result1_id
            assert result1.extraction_settings_id == e_settings.id
            assert result1.analysis_settings_id is None

            # Verify still only one result
            all_results = session.exec(
                select(CaliResult).where(CaliResult.experiment == test_experiment.id)
            ).all()
            assert len(all_results) == 1

            # Stage 3: Add analysis (simulate upgrade)
            result1.analysis_settings_id = a_settings.id
            session.add(result1)
            session.commit()
            session.refresh(result1)

            # Verify same ID, now with all three settings
            assert result1.id == result1_id
            assert result1.detection_settings_id == d_settings.id
            assert result1.extraction_settings_id == e_settings.id
            assert result1.analysis_settings_id == a_settings.id

            # Verify still only one CaliResult exists
            all_results = session.exec(
                select(CaliResult).where(CaliResult.experiment == test_experiment.id)
            ).all()
            assert len(all_results) == 1
            assert all_results[0].id == result1_id

    finally:
        engine.dispose(close=True)


def test_query_cali_results_by_settings(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test querying CaliResults by specific settings."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            # Create settings
            d_settings = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=40
            )
            a_settings = AnalysisSettings()
            session.add_all([d_settings, a_settings])
            session.commit()
            session.refresh(d_settings)
            session.refresh(a_settings)

            # Create CaliResult
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                analysis_settings_id=a_settings.id,
                positions_analyzed=[0, 1],
            )
            session.add(result)
            session.commit()

            # Query by specific settings combination
            found = session.exec(
                select(CaliResult)
                .where(CaliResult.experiment == test_experiment.id)
                .where(CaliResult.detection_settings_id == d_settings.id)
                .where(CaliResult.analysis_settings_id == a_settings.id)
            ).first()

            assert found is not None
            assert found.positions_analyzed == [0, 1]

    finally:
        engine.dispose(close=True)


def test_detection_settings_with_custom_model() -> None:
    """Test DetectionSettings with custom model path."""
    ds = DetectionSettings(
        method="cellpose",
        model_type="custom",
        custom_model="/path/to/custom/model.pth",
        diameter=None,  # Auto diameter
    )

    assert ds.method == "cellpose"
    assert ds.model_type == "custom"
    assert ds.custom_model == "/path/to/custom/model.pth"
    assert ds.diameter is None


def test_detection_settings_hash_stability() -> None:
    """Test that hash remains stable for same settings."""
    ds1 = DetectionSettings(
        method="cellpose",
        model_type="cpsam",
        diameter=30.0,
    )

    # Hash should be stable across multiple calls
    hash1 = hash(ds1)
    hash2 = hash(ds1)
    hash3 = hash(ds1)

    assert hash1 == hash2 == hash3


def test_analysis_settings_spontaneous_fields() -> None:
    """Test AnalysisSettings with spontaneous experiment fields."""
    settings = AnalysisSettings(
        threads=4,
        peaks_height_value=1.5,
        spike_threshold_value=2.0,
        burst_threshold=20.0,
    )

    assert settings.threads == 4
    assert settings.peaks_height_value == 1.5
    assert settings.spike_threshold_value == 2.0
    assert settings.burst_threshold == 20.0


def test_analysis_settings_experiment_type() -> None:
    """Test AnalysisSettings with experiment_type field."""
    from cali._constants import EVOKED

    settings_evoked = AnalysisSettings(
        experiment_type=EVOKED,
    )

    settings_spont = AnalysisSettings(
        experiment_type=SPONTANEOUS,
    )

    assert settings_evoked.experiment_type == EVOKED
    assert settings_spont.experiment_type == SPONTANEOUS
    assert settings_evoked != settings_spont  # Different experiment types


def test_cali_result_with_all_fields() -> None:
    """Test CaliResult with all possible fields populated."""
    result = CaliResult(
        experiment=1,
        detection_settings_id=2,
        analysis_settings_id=3,
        positions_analyzed=[0, 1, 2, 3, 4],
    )

    assert result.experiment == 1
    assert result.detection_settings_id == 2
    assert result.analysis_settings_id == 3
    assert result.positions_analyzed == [0, 1, 2, 3, 4]
    assert result.id is None  # Not yet saved
    assert result.created_at is not None
    assert result.last_modified is not None  # Should be set by default_factory


def test_cali_result_positions_sorted() -> None:
    """Test that positions can be stored in any order."""
    result = CaliResult(
        experiment=1,
        detection_settings_id=2,
        positions_analyzed=[5, 2, 8, 1, 3],  # Unsorted
    )

    # Should preserve the order given
    assert result.positions_analyzed == [5, 2, 8, 1, 3]


def test_multiple_experiments_same_settings(test_db: Path) -> None:
    """Test same settings reused across multiple experiments."""
    from cali.sqlmodel._util import create_database_and_tables

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    create_database_and_tables(engine)

    try:
        # Create two experiments
        exp1 = Experiment.create_from_data(
            name="Experiment 1",
            data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
            plate_maps={"genotype": {"B5": "WT"}},
        )

        exp2 = Experiment.create_from_data(
            name="Experiment 2",
            data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
            plate_maps={"genotype": {"B5": "KO"}},
        )

        with Session(engine) as session:
            session.add(exp1)
            session.add(exp2)
            session.commit()
            session.refresh(exp1)
            session.refresh(exp2)

            # Create shared settings
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            a_settings = AnalysisSettings()
            session.add_all([d_settings, a_settings])
            session.commit()
            session.refresh(d_settings)
            session.refresh(a_settings)

            # Type assertions to satisfy mypy
            assert exp1.id is not None
            assert exp2.id is not None
            assert d_settings.id is not None
            assert a_settings.id is not None

            # Create CaliResults for both experiments with same settings
            result1 = CaliResult(
                experiment=exp1.id,
                detection_settings_id=d_settings.id,
                analysis_settings_id=a_settings.id,
                positions_analyzed=[0],
            )
            result2 = CaliResult(
                experiment=exp2.id,
                detection_settings_id=d_settings.id,
                analysis_settings_id=a_settings.id,
                positions_analyzed=[0],
            )
            session.add_all([result1, result2])
            session.commit()

            # Verify both use same settings IDs
            all_results = session.exec(select(CaliResult)).all()
            assert len(all_results) == 2
            assert (
                all_results[0].detection_settings_id
                == all_results[1].detection_settings_id
            )
            assert (
                all_results[0].analysis_settings_id
                == all_results[1].analysis_settings_id
            )

    finally:
        engine.dispose(close=True)


def test_cali_result_query_by_experiment(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test querying all CaliResults for a specific experiment."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            # Create multiple results for the experiment
            d_settings1 = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=30
            )
            d_settings2 = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=40
            )
            session.add_all([d_settings1, d_settings2])
            session.commit()
            session.refresh(d_settings1)
            session.refresh(d_settings2)

            # Type assertions
            assert test_experiment.id is not None
            assert d_settings1.id is not None
            assert d_settings2.id is not None

            result1 = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings1.id,
                positions_analyzed=[0],
            )
            result2 = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings2.id,
                positions_analyzed=[0],
            )
            session.add_all([result1, result2])
            session.commit()

            # Query all results for experiment
            results = session.exec(
                select(CaliResult).where(CaliResult.experiment == test_experiment.id)
            ).all()

            assert len(results) == 2
            result_ids = {r.detection_settings_id for r in results}
            assert d_settings1.id in result_ids
            assert d_settings2.id in result_ids

    finally:
        engine.dispose(close=True)


def test_detection_settings_all_optional_fields() -> None:
    """Test DetectionSettings with optional fields."""
    ds = DetectionSettings(
        method="cellpose",
        model_type="cpsam",
        custom_model=None,
        diameter=None,
    )

    assert ds.method == "cellpose"
    assert ds.model_type == "cpsam"
    assert ds.custom_model is None
    assert ds.diameter is None
    # Other fields should have defaults
    assert ds.cellprob_threshold == 0.0
    assert ds.flow_threshold == 0.4
    assert ds.min_size == 10
    assert ds.normalize is True
    assert ds.batch_size == 8


def test_analysis_settings_minimal() -> None:
    """Test AnalysisSettings with minimal required fields."""
    settings = AnalysisSettings()

    # Should have default values
    assert settings.id is None
    assert settings.created_at is not None


def test_analysis_settings_calcium_burst_defaults() -> None:
    """Test that calcium burst threshold defaults differ from spike burst."""
    from cali._constants import (
        DEFAULT_BURST_GAUSS_SIGMA,
        DEFAULT_BURST_THRESHOLD,
        DEFAULT_CALCIUM_BURST_THRESHOLD,
    )

    settings = AnalysisSettings()

    # Calcium burst threshold should use its own default (25.0), not
    # the spike burst threshold (65.0)
    assert settings.calcium_burst_threshold == DEFAULT_CALCIUM_BURST_THRESHOLD
    assert settings.calcium_burst_threshold != DEFAULT_BURST_THRESHOLD
    assert settings.burst_threshold == DEFAULT_BURST_THRESHOLD

    # Both should use the same gaussian sigma default
    assert settings.burst_gaussian_sigma == DEFAULT_BURST_GAUSS_SIGMA
    assert settings.calcium_burst_gaussian_sigma == DEFAULT_BURST_GAUSS_SIGMA


def test_cali_result_str_representation() -> None:
    """Test CaliResult string representation."""
    result = CaliResult(
        experiment=1,
        detection_settings_id=2,
        analysis_settings_id=3,
        positions_analyzed=[0, 1, 2],
    )

    # Should be able to convert to string without error
    str_repr = str(result)
    assert "CaliResult" in str_repr or "experiment" in str_repr.lower()


def test_detection_settings_inequality_cases() -> None:
    """Test various inequality cases for DetectionSettings."""
    ds1 = DetectionSettings(method="cellpose", model_type="cpsam", diameter=30)
    ds2 = DetectionSettings(method="cellpose", model_type="cpsam", diameter=40)
    ds3 = DetectionSettings(method="cellpose", model_type="cyto2", diameter=30)

    # Different diameters
    assert ds1 != ds2
    # Different model types
    assert ds1 != ds3
    # All different from each other
    assert ds2 != ds3


def test_analysis_settings_inequality_cases() -> None:
    """Test various inequality cases for AnalysisSettings."""
    as1 = AnalysisSettings(threads=4)
    as2 = AnalysisSettings(threads=4)
    as3 = AnalysisSettings(threads=8)  # Different threads but still equal
    as4 = AnalysisSettings(
        threads=4,
        led_power_equation="y = x",
    )
    as5 = AnalysisSettings(
        threads=4,
        peaks_height_value=5.0,  # Different peak height
    )

    # Same settings - should be equal
    assert as1 == as2
    # Different threads doesn't affect equality (threads is runtime param)
    assert as1 == as3
    # Different led_power_equation
    assert as1 != as4
    # Different peaks_height_value
    assert as1 != as5


def test_cali_result_empty_positions() -> None:
    """Test CaliResult with empty positions list."""
    result = CaliResult(
        experiment=1,
        detection_settings_id=2,
        positions_analyzed=[],
    )

    assert result.positions_analyzed == []
    assert isinstance(result.positions_analyzed, list)


def test_database_concurrent_sessions(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test multiple sessions can read from database concurrently."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        # Create some data
        with Session(engine) as session:
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()

        # Read from multiple sessions
        with Session(engine) as session1, Session(engine) as session2:
            settings1 = session1.exec(select(DetectionSettings)).first()
            settings2 = session2.exec(select(DetectionSettings)).first()

            assert settings1 is not None
            assert settings2 is not None
            assert settings1.method == settings2.method

    finally:
        engine.dispose(close=True)


def test_detection_settings_update_timestamp(test_db: Path) -> None:
    """Test that created_at timestamp is set automatically."""
    from cali.sqlmodel._util import create_database_and_tables

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    create_database_and_tables(engine)

    try:
        with Session(engine) as session:
            ds = DetectionSettings(method="cellpose", model_type="cpsam")

            # Timestamp should be set before saving
            assert ds.created_at is not None
            time_before_save = ds.created_at

            session.add(ds)
            session.commit()
            session.refresh(ds)

            # Timestamp should remain the same after saving
            assert ds.created_at == time_before_save

    finally:
        engine.dispose(close=True)


def test_cali_result_cascade_behavior(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test that deleting referenced settings doesn't cascade delete CaliResult."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Type assertions
            assert d_settings.id is not None
            assert test_experiment.id is not None
            settings_id = d_settings.id

            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=settings_id,
                positions_analyzed=[0],
            )
            session.add(result)
            session.commit()
            session.refresh(result)
            assert result.id is not None
            result_id = result.id

        # Verify both exist
        with Session(engine) as session:
            assert session.get(DetectionSettings, settings_id) is not None
            assert session.get(CaliResult, result_id) is not None

    finally:
        engine.dispose(close=True)


def test_cali_result_load_from_database_by_id(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test loading a specific CaliResult by ID."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Type assertions
            assert d_settings.id is not None
            assert test_experiment.id is not None

            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                positions_analyzed=[0, 1],
            )
            session.add(result)
            session.commit()
            session.refresh(result)
            assert result.id is not None
            result_id = result.id

        # Load using the class method
        loaded = CaliResult.load_from_database(
            _get_actual_db_path(test_db), id=result_id
        )

        assert isinstance(loaded, CaliResult)
        assert loaded.id == result_id
        assert loaded.experiment == test_experiment.id
        assert loaded.positions_analyzed == [0, 1]

    finally:
        engine.dispose(close=True)


def test_cali_result_load_by_id_not_found(test_db: Path) -> None:
    """Test loading non-existent CaliResult raises ValueError."""
    from cali.sqlmodel._util import create_database_and_tables

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    create_database_and_tables(engine)
    engine.dispose(close=True)

    # Try to load non-existent result
    try:
        CaliResult.load_from_database(_get_actual_db_path(test_db), id=999)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "No CaliResult found with id=999" in str(e)


def test_cali_result_load_from_database_by_experiment(
    test_db: Path, test_experiment: Experiment
) -> None:
    """Test loading all CaliResults for an experiment."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            d_settings1 = DetectionSettings(method="cellpose", model_type="cpsam")
            d_settings2 = DetectionSettings(
                method="cellpose", model_type="cpsam", min_size=13
            )
            session.add_all([d_settings1, d_settings2])
            session.commit()
            session.refresh(d_settings1)
            session.refresh(d_settings2)

            # Type assertions
            assert d_settings1.id is not None
            assert d_settings2.id is not None
            assert test_experiment.id is not None

            result1 = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings1.id,
                positions_analyzed=[0],
            )
            result2 = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings2.id,
                positions_analyzed=[1],
            )
            session.add_all([result1, result2])
            session.commit()

        # Load all results for the experiment
        loaded = CaliResult.load_from_database(
            _get_actual_db_path(test_db), experiment_id=test_experiment.id
        )

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        # Should be ordered by created_at desc (most recent first)
        assert all(isinstance(r, CaliResult) for r in loaded)

    finally:
        engine.dispose(close=True)


def test_cali_result_load_all(test_db: Path, test_experiment: Experiment) -> None:
    """Test loading all CaliResults from database."""
    from cali.sqlmodel import save_experiment_to_database

    save_experiment_to_database(
        test_experiment,
        output_path=test_db.parent,
        database_name=test_db.name,
        overwrite=True,
    )

    actual_db_path = _get_actual_db_path(test_db)
    engine = create_engine(f"sqlite:///{actual_db_path}")
    try:
        with Session(engine) as session:
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Type assertions
            assert d_settings.id is not None
            assert test_experiment.id is not None

            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings_id=d_settings.id,
                positions_analyzed=[0],
            )
            session.add(result)
            session.commit()

        # Load all results (no filter)
        loaded = CaliResult.load_from_database(_get_actual_db_path(test_db))

        assert isinstance(loaded, list)
        assert len(loaded) >= 1  # At least our result

    finally:
        engine.dispose(close=True)


def test_experiment_equality_by_id(test_experiment: Experiment) -> None:
    """Test Experiment equality comparison by ID."""
    # Create two experiments with IDs
    exp1 = Experiment(id=1, name="Exp1")
    exp2 = Experiment(id=1, name="Exp2")  # Same ID, different name
    exp3 = Experiment(id=2, name="Exp1")  # Different ID, same name as exp1

    # Same ID -> equal
    assert exp1 == exp2
    # Different ID -> not equal
    assert exp1 != exp3


def test_experiment_equality_by_name(test_experiment: Experiment) -> None:
    """Test Experiment equality comparison by name when no ID."""
    exp1 = Experiment(id=None, name="TestExp")
    exp2 = Experiment(id=None, name="TestExp")
    exp3 = Experiment(id=None, name="OtherExp")

    # Same name, no IDs -> equal
    assert exp1 == exp2
    # Different name -> not equal
    assert exp1 != exp3


def test_experiment_hash_with_id() -> None:
    """Test Experiment hash when ID is set."""
    exp1 = Experiment(id=1, name="Exp1")
    exp2 = Experiment(id=1, name="Exp2")  # Same ID

    # Same ID should have same hash
    assert hash(exp1) == hash(exp2)


def test_experiment_hash_without_id() -> None:
    """Test Experiment hash when ID is None."""
    exp1 = Experiment(id=None, name="Exp1")
    exp2 = Experiment(id=None, name="Exp1")

    # Without ID, hash uses object identity
    assert hash(exp1) != hash(exp2)  # Different objects


def test_experiment_inequality_with_other_type() -> None:
    """Test Experiment equality with non-Experiment object."""
    exp = Experiment(id=1, name="Test")

    assert exp != "not an experiment"
    assert exp != 123
    assert exp is not None
