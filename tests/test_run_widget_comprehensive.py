"""Comprehensive tests for _RunCaliWidget using real test database."""

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot
from sqlmodel import Session, create_engine, select

from cali.gui._run_widget import _RunCaliWidget
from cali.sqlmodel._model import CaliResult, DetectionSettings, ExtractionSettings


@pytest.fixture
def test_db_path() -> Path:
    """Return path to real test database."""
    return Path("tests/test_data/data_and_db_for_tests/test_db.cali")


@pytest.fixture
def db_state(test_db_path: Path) -> dict:
    """Get the current state of the test database."""
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            # Get runs
            runs = session.exec(select(CaliResult)).all()
            run_ids = sorted([r.id for r in runs if r.id is not None])

            # Get detection settings
            detections = session.exec(select(DetectionSettings)).all()
            detection_settings = [
                (d.id, d.method) for d in detections if d.id is not None
            ]

            # Get extraction settings
            extractions = session.exec(select(ExtractionSettings)).all()
            extraction_ids = sorted([e.id for e in extractions if e.id is not None])

            return {
                "run_ids": run_ids,
                "detection_settings": detection_settings,
                "extraction_ids": extraction_ids,
                "has_runs": len(run_ids) > 0,
                "has_detections": len(detection_settings) > 0,
                "has_extractions": len(extraction_ids) > 0,
            }
    finally:
        engine.dispose(close=True)


def test_real_db_has_expected_data(db_state: dict) -> None:
    """Verify the test database has the expected data."""
    assert db_state["has_runs"], "Test database should have runs"
    assert db_state["has_detections"], "Test database should have detections"
    assert db_state["has_extractions"], "Test database should have extractions"
    assert len(db_state["run_ids"]) == 2, "Expected 2 runs"
    assert len(db_state["detection_settings"]) == 1, "Expected 1 detection setting"
    assert len(db_state["extraction_ids"]) == 2, "Expected 2 extraction settings"


def test_widget_with_empty_database(qtbot: QtBot) -> None:
    """Test widget state when no data exists (initial state)."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    from qtpy.QtCore import Qt

    model = widget._run_options_combo.model()

    # All "require" options should be disabled
    # Index 2: Extraction and Analysis (require detection)
    assert model.item(2).flags() == Qt.ItemFlag.NoItemFlags

    # Index 4: Extraction Only (require detection)
    assert model.item(4).flags() == Qt.ItemFlag.NoItemFlags

    # Index 5: Analysis Only (require detection and extraction)
    assert model.item(5).flags() == Qt.ItemFlag.NoItemFlags

    # Index 6: Export Only (require existing run)
    assert model.item(6).flags() == Qt.ItemFlag.NoItemFlags

    # Full pipeline options should be enabled
    # Index 0: Detection, Extraction and Analysis
    assert model.item(0).flags() & Qt.ItemFlag.ItemIsEnabled
    # Index 1: Detection and Extraction
    assert model.item(1).flags() & Qt.ItemFlag.ItemIsEnabled
    # Index 3: Detection Only
    assert model.item(3).flags() & Qt.ItemFlag.ItemIsEnabled


def test_widget_with_detection_only(qtbot: QtBot, db_state: dict) -> None:
    """Test widget state when only detection results exist."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Populate with detection settings only
    widget.populate_detection_settings(db_state["detection_settings"])

    from qtpy.QtCore import Qt

    model = widget._run_options_combo.model()

    # Options requiring only detection should be enabled
    # Index 2: Extraction and Analysis (require detection)
    assert model.item(2).flags() & Qt.ItemFlag.ItemIsEnabled

    # Index 4: Extraction Only (require detection)
    assert model.item(4).flags() & Qt.ItemFlag.ItemIsEnabled

    # Analysis Only should still be disabled (requires extractions too)
    # Index 5: Analysis Only (require detection and extraction)
    assert model.item(5).flags() == Qt.ItemFlag.NoItemFlags

    # Export Only should still be disabled (no runs yet)
    # Index 6: Export Only (require existing run)
    assert model.item(6).flags() == Qt.ItemFlag.NoItemFlags


def test_widget_with_detection_and_extraction(qtbot: QtBot, db_state: dict) -> None:
    """Test widget state when detection and extraction results exist."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Populate with detection and extraction settings
    widget.populate_detection_settings(db_state["detection_settings"])
    widget.populate_extraction_settings(db_state["extraction_ids"])

    from qtpy.QtCore import Qt

    model = widget._run_options_combo.model()

    # All options except Export Only should be enabled
    for i in range(6):  # 0-5
        assert model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled, (
            f"Option {i} should be enabled"
        )

    # Export Only should still be disabled (no runs yet)
    # Index 6: Export Only (require existing run)
    assert model.item(6).flags() == Qt.ItemFlag.NoItemFlags


def test_widget_with_full_database(qtbot: QtBot, db_state: dict) -> None:
    """Test widget state when all data exists (detection, extraction, runs)."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Populate with all settings
    widget.populate_detection_settings(db_state["detection_settings"])
    widget.populate_extraction_settings(db_state["extraction_ids"])
    widget.populate_run_ids(db_state["run_ids"])

    from qtpy.QtCore import Qt

    model = widget._run_options_combo.model()

    # ALL options should be enabled
    for i in range(7):  # 0-6
        assert model.item(i).flags() & Qt.ItemFlag.ItemIsEnabled, (
            f"Option {i} should be enabled with full database"
        )


def test_export_only_with_real_runs(qtbot: QtBot, db_state: dict) -> None:
    """Test Export Only option with real run IDs."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Populate with real run IDs
    widget.populate_run_ids(db_state["run_ids"])

    # Verify run IDs are populated correctly
    # Should have placeholder + 2 runs = 3 items
    assert widget._run_ids_combo.count() == len(db_state["run_ids"]) + 1

    # Check each run ID
    for i, run_id in enumerate(db_state["run_ids"], start=1):
        assert f"Run ID {run_id}" in widget._run_ids_combo.itemText(i)
        assert widget._run_ids_combo.itemData(i) == run_id


def test_detection_settings_visibility(qtbot: QtBot, db_state: dict) -> None:
    """Test detection settings combo visibility for different modes."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)
    widget.show()  # Need to show widget for visibility to work properly

    widget.populate_detection_settings(db_state["detection_settings"])
    widget.populate_extraction_settings(db_state["extraction_ids"])

    # Initially hidden
    assert not widget._detection_settings_combo.isVisible()

    # Select "Extraction Only" - should show detection combo
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(4)  # Extraction Only

    assert widget._detection_settings_combo.isVisible()

    # Select "Analysis Only" - should show detection combo
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(5)  # Analysis Only

    assert widget._detection_settings_combo.isVisible()

    # Select full pipeline - should hide detection combo
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(0)

    assert not widget._detection_settings_combo.isVisible()


def test_extraction_settings_visibility(qtbot: QtBot, db_state: dict) -> None:
    """Test extraction settings combo visibility for different modes."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)
    widget.show()  # Need to show widget for visibility to work properly

    widget.populate_detection_settings(db_state["detection_settings"])
    widget.populate_extraction_settings(db_state["extraction_ids"])

    # Initially hidden
    assert not widget._extraction_settings_combo.isVisible()

    # Only "Analysis Only" should show extraction combo
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(5)  # Analysis Only

    assert widget._extraction_settings_combo.isVisible()

    # Other modes should hide it
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(4)  # Extraction Only

    assert not widget._extraction_settings_combo.isVisible()


def test_run_ids_visibility(qtbot: QtBot, db_state: dict) -> None:
    """Test run IDs combo visibility for different modes."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)
    widget.show()  # Need to show widget for visibility to work properly

    widget.populate_run_ids(db_state["run_ids"])

    # Initially hidden
    assert not widget._run_ids_combo.isVisible()

    # Only "Export Only" should show run IDs combo
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(6)  # Export Only

    assert widget._run_ids_combo.isVisible()

    # Other modes should hide it
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(0)

    assert not widget._run_ids_combo.isVisible()


def test_value_detection_only(qtbot: QtBot, db_state: dict) -> None:
    """Test CaliRunSettings for Detection Only mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(3)  # Detection Only

    settings = widget.value()
    assert settings.run_detection
    assert not settings.run_extraction
    assert not settings.run_analysis
    assert settings.detection_settings_id is None
    assert settings.extraction_settings_id is None
    assert settings.run_id is None


def test_value_detection_and_extraction(qtbot: QtBot, db_state: dict) -> None:
    """Test CaliRunSettings for Detection and Extraction mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(1)  # Detection and Extraction

    settings = widget.value()
    assert settings.run_detection
    assert settings.run_extraction
    assert not settings.run_analysis
    assert settings.detection_settings_id is None
    assert settings.extraction_settings_id is None
    assert settings.run_id is None


def test_value_extraction_only(qtbot: QtBot, db_state: dict) -> None:
    """Test CaliRunSettings for Extraction Only mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    widget.populate_detection_settings(db_state["detection_settings"])

    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(4)  # Extraction Only

    # Select a detection ID
    widget._detection_settings_combo.setCurrentIndex(1)

    settings = widget.value()
    assert not settings.run_detection
    assert settings.run_extraction
    assert not settings.run_analysis
    assert settings.detection_settings_id == db_state["detection_settings"][0][0]
    assert settings.extraction_settings_id is None
    assert settings.run_id is None


def test_value_analysis_only(qtbot: QtBot, db_state: dict) -> None:
    """Test CaliRunSettings for Analysis Only mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    widget.populate_detection_settings(db_state["detection_settings"])
    widget.populate_extraction_settings(db_state["extraction_ids"])

    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(5)  # Analysis Only

    # Select detection and extraction IDs
    widget._detection_settings_combo.setCurrentIndex(1)
    widget._extraction_settings_combo.setCurrentIndex(1)

    settings = widget.value()
    assert not settings.run_detection
    assert not settings.run_extraction
    assert settings.run_analysis
    assert settings.detection_settings_id == db_state["detection_settings"][0][0]
    assert settings.extraction_settings_id == db_state["extraction_ids"][0]
    assert settings.run_id is None


def test_value_export_only(qtbot: QtBot, db_state: dict) -> None:
    """Test CaliRunSettings for Export Only mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    widget.populate_run_ids(db_state["run_ids"])

    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(6)  # Export Only

    # Select a run ID
    widget._run_ids_combo.setCurrentIndex(1)

    settings = widget.value()
    assert not settings.run_detection
    assert not settings.run_extraction
    assert not settings.run_analysis
    assert settings.detection_settings_id is None
    assert settings.extraction_settings_id is None
    assert settings.run_id == db_state["run_ids"][0]


def test_value_full_pipeline(qtbot: QtBot, db_state: dict) -> None:
    """Test CaliRunSettings for full pipeline mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Default is index 0 (full pipeline)
    settings = widget.value()
    assert settings.run_detection
    assert settings.run_extraction
    assert settings.run_analysis
    assert settings.detection_settings_id is None
    assert settings.extraction_settings_id is None
    assert settings.run_id is None


def test_auto_select_single_detection(qtbot: QtBot) -> None:
    """Test auto-selection when only one detection setting exists."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    widget.populate_detection_settings([(1, "cellpose")])

    # Select Extraction Only to trigger auto-selection
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(4)  # Extraction Only

    # Should auto-select the only detection available
    assert widget._detection_settings_combo.currentData() == 1


def test_auto_select_single_extraction(qtbot: QtBot, db_state: dict) -> None:
    """Test auto-selection when only one extraction setting exists."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    widget.populate_detection_settings(db_state["detection_settings"])
    widget.populate_extraction_settings([1])  # Only one extraction

    # Select Analysis Only to trigger auto-selection
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(5)  # Analysis Only

    # Should auto-select the only extraction available
    assert widget._extraction_settings_combo.currentData() == 1


def test_auto_select_single_run(qtbot: QtBot) -> None:
    """Test auto-selection when only one run exists."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    widget.populate_run_ids([42])

    # Select Export Only to trigger auto-selection
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(6)  # Export Only

    # Should auto-select the only run available
    assert widget._run_ids_combo.currentData() == 42


def test_disabled_option_reverts_to_default(qtbot: QtBot, db_state: dict) -> None:
    """Test that disabled options revert to default when database changes."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Start with full database
    widget.populate_detection_settings(db_state["detection_settings"])
    widget.populate_extraction_settings(db_state["extraction_ids"])

    # Select Analysis Only
    with qtbot.waitSignal(widget._run_options_combo.currentTextChanged, timeout=1000):
        widget._run_options_combo.setCurrentIndex(5)  # Analysis Only

    # Clear extraction settings (simulating database change)
    widget.populate_extraction_settings([])

    # Should revert to default (index 0)
    assert widget._run_options_combo.currentIndex() == 0
