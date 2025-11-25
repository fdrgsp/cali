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
)


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
        experiment_type=SPONTANEOUS,
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
        dff_window=100,
        peaks_height_value=1.5,
        spike_threshold_value=2.0,
        threads=4,
        neuropil_inner_radius=5,
        burst_threshold=20.0,
    )

    time.sleep(0.001)  # Ensure different timestamps

    as2 = AnalysisSettings(
        dff_window=100,
        peaks_height_value=1.5,
        spike_threshold_value=2.0,
        threads=4,
        neuropil_inner_radius=5,
        burst_threshold=20.0,
    )

    # Should be equal despite different created_at times
    assert as1.created_at != as2.created_at, "Should have different timestamps"
    assert as1 == as2, "AnalysisSettings with same parameters should be equal"
    assert hash(as1) == hash(as2), "Equal AnalysisSettings should have same hash"

    # Different settings should not be equal
    as3 = AnalysisSettings(
        dff_window=200,  # Different window
        peaks_height_value=1.5,
        spike_threshold_value=2.0,
        threads=4,
    )
    assert as1 != as3, "AnalysisSettings with different parameters should not be equal"


def test_cali_result_equality() -> None:
    """Test CaliResult equality ignores created_at timestamp."""
    import time

    result1 = CaliResult(experiment=1, analysis_settings=1, positions_analyzed=[0, 1])

    time.sleep(0.001)  # Ensure different timestamps

    result2 = CaliResult(experiment=1, analysis_settings=1, positions_analyzed=[0, 1])

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
        analysis_settings=2,  # Different analysis settings
        positions_analyzed=[0, 1],
    )

    assert result1 != result3, "CaliResults with different settings should not be equal"

    # Test hash consistency
    assert hash(result1) == hash(result2), "Equal CaliResults should have same hash"


def test_cali_result_with_none_values() -> None:
    """Test CaliResult equality with None values."""
    result1 = CaliResult(
        experiment=1,
        detection_settings=None,
        analysis_settings=1,
        positions_analyzed=None,
    )

    result2 = CaliResult(
        experiment=1,
        detection_settings=None,
        analysis_settings=1,
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

    engine = create_engine(f"sqlite:///{test_db}")
    try:
        with Session(engine) as session:
            # Create detection settings
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Create analysis settings
            a_settings = AnalysisSettings(dff_window=100)
            session.add(a_settings)
            session.commit()
            session.refresh(a_settings)

            # Create CaliResult
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings=d_settings.id,
                analysis_settings=a_settings.id,
                positions_analyzed=[0, 1, 2],
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            # Verify it was saved
            loaded_result = session.get(CaliResult, result.id)
            assert loaded_result is not None
            assert loaded_result.experiment == test_experiment.id
            assert loaded_result.detection_settings == d_settings.id
            assert loaded_result.analysis_settings == a_settings.id
            assert loaded_result.positions_analyzed == [0, 1, 2]

    finally:
        engine.dispose(close=True)


def test_detection_settings_database_deduplication(test_db: Path) -> None:
    """Test that identical DetectionSettings are deduplicated in database."""
    from cali.sqlmodel._util import create_database_and_tables

    engine = create_engine(f"sqlite:///{test_db}")
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

    engine = create_engine(f"sqlite:///{test_db}")
    create_database_and_tables(engine)

    try:
        with Session(engine) as session:
            # Create first analysis settings
            as1 = AnalysisSettings(dff_window=100, threads=4)
            session.add(as1)
            session.commit()

        with Session(engine) as session:
            # Try to create identical settings
            as2 = AnalysisSettings(dff_window=100, threads=4)

            # Check if it already exists
            existing = session.exec(
                select(AnalysisSettings).where(
                    AnalysisSettings.dff_window == as2.dff_window,
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
        dff_window=100,
        led_power_equation="y = 0.5 * x",
        led_pulse_duration=50.0,
        led_pulse_powers=[5.0, 10.0, 15.0],
        led_pulse_on_frames=[100, 200, 300],
    )

    assert settings.dff_window == 100
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

    engine = create_engine(f"sqlite:///{test_db}")
    try:
        with Session(engine) as session:
            # Create settings
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            a_settings = AnalysisSettings(dff_window=100)
            session.add(d_settings)
            session.add(a_settings)
            session.commit()
            session.refresh(d_settings)
            session.refresh(a_settings)

            # Create CaliResult linking to both
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings=d_settings.id,
                analysis_settings=a_settings.id,
                positions_analyzed=[0],
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            # Verify linkages
            assert result.detection_settings == d_settings.id
            assert result.analysis_settings == a_settings.id

            # Verify we can query back
            loaded_d_settings = session.get(
                DetectionSettings, result.detection_settings
            )
            loaded_a_settings = session.get(AnalysisSettings, result.analysis_settings)

            assert loaded_d_settings is not None
            assert loaded_a_settings is not None
            assert loaded_d_settings.method == "cellpose"
            assert loaded_a_settings.dff_window == 100

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

    engine = create_engine(f"sqlite:///{test_db}")
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
                detection_settings=d_settings.id,
                analysis_settings=None,  # No analysis
                positions_analyzed=[0],
            )
            session.add(result)
            session.commit()
            session.refresh(result)

            # Verify
            assert result.detection_settings == d_settings.id
            assert result.analysis_settings is None

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

    engine = create_engine(f"sqlite:///{test_db}")
    try:
        with Session(engine) as session:
            d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
            session.add(d_settings)
            session.commit()
            session.refresh(d_settings)

            # Create with specific positions
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings=d_settings.id,
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

    engine = create_engine(f"sqlite:///{test_db}")
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
            a_settings_1 = AnalysisSettings(dff_window=100)
            a_settings_2 = AnalysisSettings(dff_window=200)
            session.add_all([a_settings_1, a_settings_2])
            session.commit()
            session.refresh(a_settings_1)
            session.refresh(a_settings_2)

            # Create multiple CaliResults
            result_1 = CaliResult(
                experiment=test_experiment.id,
                detection_settings=d_settings_1.id,
                analysis_settings=a_settings_1.id,
                positions_analyzed=[0],
            )
            result_2 = CaliResult(
                experiment=test_experiment.id,
                detection_settings=d_settings_2.id,
                analysis_settings=a_settings_2.id,
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
                (r.detection_settings, r.analysis_settings) for r in all_results
            }
            assert len(settings_pairs) == 2

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

    engine = create_engine(f"sqlite:///{test_db}")
    try:
        with Session(engine) as session:
            # Create settings
            d_settings = DetectionSettings(
                method="cellpose", model_type="cpsam", diameter=40
            )
            a_settings = AnalysisSettings(dff_window=150)
            session.add_all([d_settings, a_settings])
            session.commit()
            session.refresh(d_settings)
            session.refresh(a_settings)

            # Create CaliResult
            result = CaliResult(
                experiment=test_experiment.id,
                detection_settings=d_settings.id,
                analysis_settings=a_settings.id,
                positions_analyzed=[0, 1],
            )
            session.add(result)
            session.commit()

            # Query by specific settings combination
            found = session.exec(
                select(CaliResult)
                .where(CaliResult.experiment == test_experiment.id)
                .where(CaliResult.detection_settings == d_settings.id)
                .where(CaliResult.analysis_settings == a_settings.id)
            ).first()

            assert found is not None
            assert found.positions_analyzed == [0, 1]

    finally:
        engine.dispose(close=True)
