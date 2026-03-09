"""Tests for RunsPanel.get_run_ids() method."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cali.gui._runs_panel import _RunsPanel

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    pass
from cali.sqlmodel import CaliResult, DetectionSettings, ExtractionSettings


@pytest.fixture
def test_db_path() -> Path:
    """Return path to test database."""
    return Path("tests/test_data/data_and_db_for_tests/test_db.cali")


@pytest.fixture
def test_data_path() -> Path:
    """Return path to test data directory."""
    return Path("tests/test_data/data_and_db_for_tests")


def test_get_run_ids_no_database(qtbot: QtBot) -> None:
    """Test get_run_ids returns empty list when no database is loaded."""
    panel = _RunsPanel()
    qtbot.addWidget(panel)

    result = panel.get_run_ids()
    assert result == []


def test_get_run_ids_with_database(
    qtbot: QtBot,
    test_db_path: Path,
    test_data_path: Path,
) -> None:
    """Test get_run_ids returns sorted list of run IDs from database."""
    panel = _RunsPanel()
    qtbot.addWidget(panel)

    # Set database path
    panel._database_path = str(test_db_path)

    result = panel.get_run_ids()

    # Should return a sorted list of integers
    assert isinstance(result, list)
    assert all(isinstance(_id, int) for _id in result)
    assert result == sorted(result)  # Should be sorted

    # Explicitly clean up
    panel._database_path = None
    panel.clear()
    qtbot.wait(50)
    gc.collect()  # Force garbage collection to clean up connections


def test_get_run_ids_handles_exception(qtbot: QtBot, tmp_path: Path) -> None:
    """Test get_run_ids handles database errors gracefully."""
    panel = _RunsPanel()
    qtbot.addWidget(panel)

    # Set invalid database path
    invalid_db = tmp_path / "nonexistent.cali"
    panel._database_path = str(invalid_db)

    # Should return empty list on error, not raise exception
    result = panel.get_run_ids()
    assert result == []


def test_get_run_ids_filters_none_values(
    qtbot: QtBot,
    test_db_path: Path,
) -> None:
    """Test get_run_ids filters out None values from results."""
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel._database_path = str(test_db_path)

    # Get actual run IDs
    result = panel.get_run_ids()

    # Should not contain None
    assert None not in result
    # All values should be integers
    assert all(isinstance(_id, int) for _id in result)

    # Explicitly clean up
    panel._database_path = None
    panel.clear()
    qtbot.wait(50)
    gc.collect()  # Force garbage collection to clean up connections


# ============================================================================
# Incomplete Run Indicator Tests
# ============================================================================


@pytest.fixture
def runs_panel(qtbot: QtBot) -> _RunsPanel:
    """Create a runs panel widget."""
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    return panel


def test_incomplete_extraction_shows_asterisk(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    """Test that incomplete extraction shows asterisk in runs panel."""
    from sqlmodel import Session, create_engine

    # Create database
    db_path = tmp_path / "test.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create schema
    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    # Create test data
    with Session(engine) as session:
        # Detection settings
        d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d_settings)
        session.flush()
        detection_id = d_settings.id

        # Extraction settings
        e_settings = ExtractionSettings(neuropil_inner_radius=10)
        session.add(e_settings)
        session.flush()
        extraction_id = e_settings.id

        # Create result with incomplete extraction
        # Detected [0, 1, 2] but only extracted [0, 1]
        result = CaliResult(
            experiment=1,
            detection_settings_id=detection_id,
            extraction_settings_id=extraction_id,
            analysis_settings_id=None,
            positions_detected=[0, 1, 2],
            positions_extracted=[0, 1],  # Incomplete!
            positions_analyzed=None,
        )
        session.add(result)
        session.commit()

    engine.dispose(close=True)

    # Load runs panel
    runs_panel.set_database_path(db_path)

    # Wait for UI to update
    qtbot.wait(100)

    # Check that the item text contains asterisk for extraction
    assert runs_panel._runs_list.count() == 1
    item = runs_panel._runs_list.item(0)
    assert item is not None
    item_text = item.text()

    # Should have asterisk next to Extraction ID
    assert f"Extraction ID: {extraction_id} ⚠️" in item_text
    # Should NOT have asterisk next to Analysis (it's None)
    assert "Analysis ID: None ⚠️" not in item_text

    # Explicitly clear to help with cleanup
    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()  # Force garbage collection


def test_incomplete_analysis_shows_asterisk(
    tmp_path: Path, runs_panel: _RunsPanel
) -> None:
    """Test that incomplete analysis shows asterisk in runs panel."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel import AnalysisSettings

    # Create database
    db_path = tmp_path / "test.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create schema
    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    # Create test data
    with Session(engine) as session:
        # Settings
        d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d_settings)
        session.flush()
        detection_id = d_settings.id

        e_settings = ExtractionSettings(neuropil_inner_radius=10)
        session.add(e_settings)
        session.flush()
        extraction_id = e_settings.id

        a_settings = AnalysisSettings(peaks_height_value=2.0)
        session.add(a_settings)
        session.flush()
        analysis_id = a_settings.id

        # Create result with incomplete analysis
        # Extracted [0, 1, 2] but only analyzed [0, 1]
        result = CaliResult(
            experiment=1,
            detection_settings_id=detection_id,
            extraction_settings_id=extraction_id,
            analysis_settings_id=analysis_id,
            positions_detected=[0, 1, 2],
            positions_extracted=[0, 1, 2],
            positions_analyzed=[0, 1],  # Incomplete!
        )
        session.add(result)
        session.commit()

    engine.dispose(close=True)

    # Load runs panel
    runs_panel.set_database_path(db_path)

    # Check that the item text contains asterisk for analysis
    assert runs_panel._runs_list.count() == 1
    item = runs_panel._runs_list.item(0)
    assert item is not None
    item_text = item.text()

    # Should NOT have asterisk next to Extraction (it's complete)
    assert f"Extraction ID: {extraction_id} ⚠️" not in item_text
    assert f"Extraction ID: {extraction_id}\n" in item_text
    # Should have asterisk next to Analysis ID
    assert f"Analysis ID: {analysis_id} ⚠️" in item_text


def test_complete_run_no_asterisk(tmp_path: Path, runs_panel: _RunsPanel) -> None:
    """Test that complete runs don't show asterisk."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel import AnalysisSettings

    # Create database
    db_path = tmp_path / "test.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create schema
    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    # Create test data
    with Session(engine) as session:
        # Settings
        d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d_settings)
        session.flush()
        detection_id = d_settings.id

        e_settings = ExtractionSettings(neuropil_inner_radius=10)
        session.add(e_settings)
        session.flush()
        extraction_id = e_settings.id

        a_settings = AnalysisSettings(peaks_height_value=2.0)
        session.add(a_settings)
        session.flush()
        analysis_id = a_settings.id

        # Create result with complete pipeline
        result = CaliResult(
            experiment=1,
            detection_settings_id=detection_id,
            extraction_settings_id=extraction_id,
            analysis_settings_id=analysis_id,
            positions_detected=[0, 1, 2],
            positions_extracted=[0, 1, 2],  # Complete
            positions_analyzed=[0, 1, 2],  # Complete
        )
        session.add(result)
        session.commit()

    engine.dispose(close=True)

    # Load runs panel
    runs_panel.set_database_path(db_path)

    # Check that the item text does NOT contain asterisks
    assert runs_panel._runs_list.count() == 1
    item = runs_panel._runs_list.item(0)
    assert item is not None
    item_text = item.text()

    # Should NOT have asterisks
    assert f"Extraction ID: {extraction_id} ⚠️" not in item_text
    assert f"Analysis ID: {analysis_id} ⚠️" not in item_text
    # But should have the IDs
    assert f"Extraction ID: {extraction_id}" in item_text
    assert f"Analysis ID: {analysis_id}" in item_text


def test_detection_only_run_no_asterisk(tmp_path: Path, runs_panel: _RunsPanel) -> None:
    """Test that detection-only runs don't show asterisk."""
    from sqlmodel import Session, create_engine

    # Create database
    db_path = tmp_path / "test.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create schema
    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    # Create test data
    with Session(engine) as session:
        # Detection settings
        d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d_settings)
        session.flush()

        # Create detection-only result
        result = CaliResult(
            experiment=1,
            detection_settings_id=d_settings.id,
            extraction_settings_id=None,
            analysis_settings_id=None,
            positions_detected=[0, 1, 2],
            positions_extracted=None,
            positions_analyzed=None,
        )
        session.add(result)
        session.commit()

    engine.dispose(close=True)

    # Load runs panel
    runs_panel.set_database_path(db_path)

    # Check that the item text does NOT contain asterisks
    assert runs_panel._runs_list.count() == 1
    item = runs_panel._runs_list.item(0)
    assert item is not None
    item_text = item.text()

    # Should NOT have asterisks (extraction/analysis are None, not incomplete)
    assert " ⚠️" not in item_text
    # Should have ❌ for extraction and analysis
    assert "❌ Extraction ID: None" in item_text
    assert "❌ Analysis ID: None" in item_text


def test_both_incomplete_shows_both_asterisks(
    tmp_path: Path, runs_panel: _RunsPanel
) -> None:
    """Test that both extraction and analysis can show asterisks simultaneously."""
    from sqlmodel import Session, create_engine

    from cali.sqlmodel import AnalysisSettings

    # Create database
    db_path = tmp_path / "test.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create schema
    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    # Create test data
    with Session(engine) as session:
        # Settings
        d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d_settings)
        session.flush()
        detection_id = d_settings.id

        e_settings = ExtractionSettings(neuropil_inner_radius=10)
        session.add(e_settings)
        session.flush()
        extraction_id = e_settings.id

        a_settings = AnalysisSettings(peaks_height_value=2.0)
        session.add(a_settings)
        session.flush()
        analysis_id = a_settings.id

        # Create result with both incomplete
        # Detected [0, 1, 2, 3], extracted [0, 1, 2], analyzed [0, 1]
        result = CaliResult(
            experiment=1,
            detection_settings_id=detection_id,
            extraction_settings_id=extraction_id,
            analysis_settings_id=analysis_id,
            positions_detected=[0, 1, 2, 3],
            positions_extracted=[0, 1, 2],  # Incomplete!
            positions_analyzed=[0, 1],  # Incomplete!
        )
        session.add(result)
        session.commit()

    engine.dispose(close=True)

    # Load runs panel
    runs_panel.set_database_path(db_path)

    # Check that both have asterisks
    assert runs_panel._runs_list.count() == 1
    item = runs_panel._runs_list.item(0)
    assert item is not None
    item_text = item.text()

    # Should have asterisks for both
    assert f"Extraction ID: {extraction_id} ⚠️" in item_text
    assert f"Analysis ID: {analysis_id} ⚠️" in item_text
