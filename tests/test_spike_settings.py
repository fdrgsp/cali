"""Method-bound settings, exact legacy backfill, and execution gating."""

import math
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cali.analysis._analysis_runner import AnalysisRunner
from cali.extraction._extraction_runner import ExtractionRunner
from cali.sqlmodel import (
    AnalysisSettings,
    ExtractionSettings,
    SpikeAnalysisSettings,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._spike_settings import LEGACY_SPIKE_SETTING_NAMES


@pytest.mark.parametrize(
    "constructor", [ExtractionSettings, ExtractionSettings.model_validate]
)
def test_canonical_output_selection(constructor: object) -> None:
    data = {
        "spike_methods": ["cascade", "oasis", "cascade"],
        "cascade_model": " explicit-model ",
        "cascade_device": "cpu",
    }
    settings = (
        constructor(data)
        if constructor == ExtractionSettings.model_validate
        else constructor(**data)
    )
    assert settings.spike_methods == ("oasis", "cascade")
    assert settings.cascade_model == "explicit-model"
    assert settings.cascade_device == "cpu"
    assert settings == ExtractionSettings(
        spike_methods=("oasis", "cascade"),
        cascade_model="explicit-model",
        cascade_device="cpu",
    )
    assert hash(settings) == hash(ExtractionSettings(**data))


@pytest.mark.parametrize("methods", [[], (), ["unknown"], "oasis", None, {"oasis"}])
def test_invalid_method_selection(methods: object) -> None:
    with pytest.raises(ValueError, match="spike_methods"):
        ExtractionSettings(spike_methods=methods)
    with pytest.raises(ValueError, match="spike_methods"):
        ExtractionSettings.model_validate({"spike_methods": methods})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cascade_model": None},
        {"cascade_model": " "},
        {"cascade_model": "model", "cascade_device": "gpu"},
    ],
)
def test_cascade_requires_explicit_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExtractionSettings(spike_methods=("cascade",), **kwargs)


def test_oasis_clears_unused_cascade_inputs() -> None:
    settings = ExtractionSettings(cascade_model="ignored", cascade_device="invalid")
    assert settings.cascade_model is None
    assert settings.cascade_device == "auto"
    assert settings == ExtractionSettings()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"discard_initial_value": -1},
        {"discard_initial_value": math.nan},
        {"discard_initial_value": math.inf},
        {"discard_initial_value": 1.5},
        {"discard_initial_unit": "minutes"},
    ],
)
def test_discard_settings_validate_at_construction(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExtractionSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spike_methods": ("cascade",), "cascade_model": "model"},
        {"spike_methods": ("oasis", "cascade"), "cascade_model": "model"},
        {"discard_initial_value": 2},
        {"discard_initial_unit": "seconds"},
        {"frame_rate_verified": True},
    ],
)
def test_output_settings_change_semantic_identity(kwargs: dict) -> None:
    assert ExtractionSettings(**kwargs) != ExtractionSettings()
    assert hash(ExtractionSettings(**kwargs)) != hash(ExtractionSettings())


def test_cascade_model_and_device_change_identity() -> None:
    base = {"spike_methods": ("cascade",), "cascade_model": "one"}
    settings = ExtractionSettings(**base)
    assert settings != ExtractionSettings(**{**base, "cascade_model": "two"})
    assert settings != ExtractionSettings(**base, cascade_device="cpu")


def test_method_specific_defaults_and_legal_thresholds() -> None:
    settings = AnalysisSettings()
    oasis = settings.get_spike_settings("oasis")
    assert oasis.threshold_mode == "multiplier"
    assert oasis.threshold_value == 3
    cascade = SpikeAnalysisSettings(method="cascade")
    assert cascade.threshold_mode == "cascade_ap"
    assert cascade.threshold_value is None
    assert cascade.cascade_ap_threshold_fraction == 1 / math.e
    assert (
        SpikeAnalysisSettings.model_validate({"method": "cascade"}).semantic_key()
        == cascade.semantic_key()
    )
    global_cascade = SpikeAnalysisSettings(
        method="cascade", threshold_mode="global", threshold_value=0.02
    )
    assert global_cascade.threshold_value == 0.02
    assert global_cascade.cascade_ap_threshold_fraction is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "unknown"},
        {"method": "oasis", "threshold_mode": "cascade_ap"},
        {"method": "cascade", "threshold_mode": "multiplier"},
        {"method": "cascade", "threshold_mode": "global"},
        {"threshold_value": math.inf},
        {"threshold_value": -1},
        {"method": "cascade", "cascade_ap_threshold_fraction": 0},
        {"method": "cascade", "cascade_ap_threshold_fraction": 1.1},
    ],
)
def test_invalid_method_thresholds(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SpikeAnalysisSettings(**kwargs)
    with pytest.raises(ValueError):
        SpikeAnalysisSettings.model_validate(kwargs)


def test_children_control_equality_independent_of_order() -> None:
    first = AnalysisSettings(
        spike_settings=[
            SpikeAnalysisSettings(),
            SpikeAnalysisSettings(method="cascade"),
        ]
    )
    second = AnalysisSettings(
        spike_settings=[
            SpikeAnalysisSettings(method="cascade"),
            SpikeAnalysisSettings(),
        ]
    )
    assert first == second
    assert hash(first) == hash(second)
    second.get_spike_settings("cascade").burst_threshold += 1
    assert first != second
    assert hash(first) != hash(second)
    with pytest.raises(ValueError, match="Duplicate"):
        AnalysisSettings(
            spike_settings=[SpikeAnalysisSettings(), SpikeAnalysisSettings()]
        )
    with pytest.raises(ValueError, match="match"):
        first.validate_spike_settings(("oasis",))


def test_legacy_settings_are_oasis_only_compatibility_accessors() -> None:
    settings = AnalysisSettings(
        spike_threshold_value=1.25, spikes_sync_jitter_window=321
    )
    settings.spike_threshold_value = 2.5
    assert settings.get_spike_settings("oasis").threshold_value == 2.5
    assert settings.get_spike_settings("oasis").spikes_sync_jitter_window == 321
    settings.get_spike_settings("oasis").threshold_value = 4
    assert settings.spike_threshold_value == 4
    assert AnalysisSettings.model_validate(settings.model_dump()) == settings
    assert AnalysisSettings.model_validate_json(settings.model_dump_json()) == settings
    with pytest.raises(ValueError, match="exactly one"):
        _ = AnalysisSettings(
            spike_settings=[SpikeAnalysisSettings(method="cascade")]
        ).spike_threshold_value


def test_validation_does_not_reparent_source_children() -> None:
    settings = AnalysisSettings(spike_threshold_value=1.1)
    child = settings.get_spike_settings("oasis")
    validated = AnalysisSettings.model_validate(settings)
    assert validated == settings
    assert settings.get_spike_settings("oasis") is child
    assert child.analysis_settings is settings
    assert validated.get_spike_settings("oasis") is not child
    validated.spike_threshold_value = 2
    assert settings.spike_threshold_value == 1.1


def _legacy_settings_database(path: Path) -> dict:
    """Build a genuine v2 settings table without normalized children."""
    values = AnalysisSettings(
        spike_threshold_value=1.234,
        spike_threshold_mode="global",
        burst_threshold=73.5,
        burst_min_duration=777,
        burst_gaussian_sigma=0.9,
        spikes_sync_cross_corr_lag=432,
        spikes_sync_jitter_window=123,
        ccg_n_shuffles=7,
        enable_rising_edge_analysis=True,
    )
    expected = {name: getattr(values, name) for name in LEGACY_SPIKE_SETTING_NAMES}
    engine = create_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        session.add(values)
        session.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE spike_analysis_settings")
        for name in ("spike_methods", "cascade_model", "cascade_device"):
            connection.exec_driver_sql(
                f"ALTER TABLE extraction_settings DROP COLUMN {name}"
            )
        assignments = ", ".join(f"{name} = ?" for name in expected)
        connection.exec_driver_sql(
            f"UPDATE analysis_settings SET {assignments} WHERE id = 1",
            tuple(expected.values()),
        )
        connection.exec_driver_sql("PRAGMA user_version = 2")
    engine.dispose()
    return expected


def test_exact_backfill_detachment_and_legacy_columns_remain_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.cali"
    expected = _legacy_settings_database(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        detached = AnalysisSettings.load_from_database(path, id=1)
        for old, new in LEGACY_SPIKE_SETTING_NAMES.items():
            assert getattr(detached.get_spike_settings("oasis"), new) == expected[old]
        with Session(engine) as session:
            loaded = session.get(AnalysisSettings, 1)
            loaded.spike_threshold_value = 6
            session.add(AnalysisSettings(spike_threshold_value=9))
            session.commit()
        with engine.connect() as connection:
            assert (
                connection.exec_driver_sql(
                    "SELECT spike_threshold_value FROM analysis_settings WHERE id = 1"
                ).scalar_one()
                == expected["spike_threshold_value"]
            )
        ensure_schema_current(engine)
        assert (
            AnalysisSettings.load_from_database(path, id=1).spike_threshold_value == 6
        )
        with Session(engine) as session:
            assert len(session.exec(select(SpikeAnalysisSettings)).all()) == 2
    finally:
        engine.dispose()


def test_settings_migration_rolls_back_ddl_and_backfill(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    expected = _legacy_settings_database(path)
    engine = create_engine(f"sqlite:///{path}")
    migration = _engine._method_settings

    def interrupted(connection: object) -> None:
        migration(connection)
        raise RuntimeError("settings interruption")

    try:
        with patch.object(
            _engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:2], interrupted)
        ):
            with pytest.raises(RuntimeError, match="interruption"):
                ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 2
            columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(extraction_settings)"
                )
            }
            assert "spike_methods" not in columns
            assert (
                connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master "
                    "WHERE name = 'spike_analysis_settings'"
                ).all()
                == []
            )
        ensure_schema_current(engine)
        assert (
            AnalysisSettings.load_from_database(path, id=1).spike_threshold_value
            == expected["spike_threshold_value"]
        )
    finally:
        engine.dispose()


def test_conflicting_partial_backfill_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "conflicting.cali"
    _legacy_settings_database(path)
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            _engine._method_settings(connection)
            connection.exec_driver_sql(
                "UPDATE spike_analysis_settings SET threshold_value = 999"
            )
        with pytest.raises(ValueError, match="exact-value"):
            ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 2
    finally:
        engine.dispose()


def test_dual_settings_roundtrip_and_unique_constraint(tmp_path: Path) -> None:
    path = tmp_path / "dual.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            session.add(
                ExtractionSettings(
                    spike_methods=["cascade", "oasis"],
                    cascade_model="model",
                    cascade_device="cpu",
                )
            )
            session.add(
                AnalysisSettings(
                    spike_settings=[
                        SpikeAnalysisSettings(),
                        SpikeAnalysisSettings(method="cascade"),
                    ]
                )
            )
            session.commit()
        extraction = ExtractionSettings.load_from_database(path, id=1)
        analysis = AnalysisSettings.load_from_database(path, id=1)
        assert extraction.spike_methods == ("oasis", "cascade")
        assert isinstance(extraction.spike_methods, tuple)
        assert analysis.get_spike_settings("cascade").threshold_mode == "cascade_ap"
        with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                connection.exec_driver_sql(
                    "INSERT INTO spike_analysis_settings SELECT NULL, "
                    "analysis_settings_id, method, threshold_mode, threshold_value, "
                    "cascade_ap_threshold_fraction, burst_threshold, "
                    "burst_min_duration, burst_gaussian_sigma, "
                    "spikes_sync_cross_corr_lag, spikes_sync_jitter_window, "
                    "ccg_n_shuffles, enable_rising_edge_analysis "
                    "FROM spike_analysis_settings WHERE method = 'oasis'"
                )
    finally:
        engine.dispose()


def test_mutations_are_validated_before_database_write() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            settings = ExtractionSettings(
                spike_methods=("cascade",), cascade_model="model"
            )
            settings.cascade_model = None
            session.add(settings)
            with pytest.raises(ValueError, match="cascade_model"):
                session.commit()
            session.rollback()
    finally:
        engine.dispose()


def test_cascade_spike_analysis_rejected_before_computation() -> None:
    extraction = ExtractionSettings(spike_methods=("cascade",), cascade_model="model")
    with pytest.raises(NotImplementedError, match="not available"):
        ExtractionRunner().run(
            None,
            extraction,
            [],
            analysis_settings=AnalysisSettings(
                spike_settings=[SpikeAnalysisSettings(method="cascade")]
            ),
        )
    analysis = AnalysisSettings(
        spike_settings=[SpikeAnalysisSettings(method="cascade")]
    )
    with pytest.raises(NotImplementedError, match="not available"):
        AnalysisRunner().run([], analysis)
