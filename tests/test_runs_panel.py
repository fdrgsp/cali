"""Tests for _RunsPanel — runs list, saved-segmentations, and keep-detection flows."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QListWidgetItem, QMessageBox

from cali.gui._runs_panel import _DetectionKeepDialog, _DetectionSummary, _RunsPanel
from cali.sqlmodel import CaliResult, DetectionSettings, ExtractionSettings
from cali.sqlmodel._model import SQLModel

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


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


# ============================================================================
# Helpers
# ============================================================================


def _make_db(tmp_path: Path) -> Path:
    """Create a fresh SQLite database with the cali schema."""
    from sqlmodel import create_engine

    db_path = tmp_path / "test.cali"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    engine.dispose(close=True)
    return db_path


def _add_detection_and_run(
    session: object, method: str = "cellpose", model_type: str = "cpsam"
) -> tuple[int, int]:
    """Add a DetectionSettings + CaliResult pair; return (detection_id, run_id)."""
    d = DetectionSettings(method=method, model_type=model_type)
    session.add(d)  # type: ignore[union-attr]
    session.flush()  # type: ignore[union-attr]
    r = CaliResult(experiment=1, detection_settings_id=d.id, positions_detected=[0])
    session.add(r)  # type: ignore[union-attr]
    session.flush()  # type: ignore[union-attr]
    assert d.id is not None and r.id is not None
    return d.id, r.id


# ============================================================================
# _DetectionSummary.label()
# ============================================================================


@pytest.mark.parametrize(
    "run_count,roi_count,fov_count,model_type,expected_snippets",
    [
        (1, 1, 1, "cpsam", ["1 run", "1 ROI", "1 FOV", "cpsam"]),
        (2, 5, 3, "cyto3", ["2 runs", "5 ROIs", "3 FOVs", "cyto3"]),
        (1, 0, 0, None, ["1 run", "0 ROIs", "0 FOVs"]),
    ],
)
def test_detection_summary_label(
    run_count: int,
    roi_count: int,
    fov_count: int,
    model_type: str | None,
    expected_snippets: list[str],
) -> None:
    s = _DetectionSummary(
        detection_id=42,
        method="cellpose",
        model_type=model_type,
        run_count=run_count,
        roi_count=roi_count,
        fov_count=fov_count,
    )
    label = s.label()
    for snippet in expected_snippets:
        assert snippet in label


def test_detection_summary_label_no_model_type_omits_slash() -> None:
    s = _DetectionSummary(42, "cellpose", None, 1, 0, 0)
    label = s.label()
    assert "cellpose" in label
    # No model_type → no " / " separator
    assert " / " not in label


# ============================================================================
# _DetectionKeepDialog
# ============================================================================


def test_detection_keep_dialog_with_summaries(qtbot: QtBot) -> None:
    summaries = [
        _DetectionSummary(1, "cellpose", "cpsam", 2, 5, 2),
        _DetectionSummary(2, "cellpose", "cyto3", 1, 3, 1),
    ]
    dialog = _DetectionKeepDialog(summaries)
    qtbot.addWidget(dialog)
    assert set(dialog._checkboxes) == {1, 2}
    assert dialog.kept_detection_ids() == set()  # nothing checked by default


def test_detection_keep_dialog_no_summaries(qtbot: QtBot) -> None:
    dialog = _DetectionKeepDialog([])
    qtbot.addWidget(dialog)
    assert dialog._checkboxes == {}
    assert dialog.kept_detection_ids() == set()


@pytest.mark.parametrize("checked_ids", [{1}, {2}, {1, 2}])
def test_detection_keep_dialog_kept_ids(qtbot: QtBot, checked_ids: set) -> None:
    summaries = [
        _DetectionSummary(1, "cellpose", "cpsam", 2, 5, 2),
        _DetectionSummary(2, "cellpose", "cyto3", 1, 3, 1),
    ]
    dialog = _DetectionKeepDialog(summaries)
    qtbot.addWidget(dialog)
    for did in checked_ids:
        dialog._checkboxes[did].setChecked(True)
    assert dialog.kept_detection_ids() == checked_ids


# ============================================================================
# refresh_runs() — saved-segmentations list
# ============================================================================


def test_refresh_runs_orphan_detection_appears_in_saved_list(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    """DetectionSettings with no CaliResult should show in the saved-segs list."""
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        orphan = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(orphan)
        session.commit()
        orphan_id = orphan.id
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    assert runs_panel._runs_list.count() == 0
    assert runs_panel._saved_segs_list.count() == 1
    item = runs_panel._saved_segs_list.item(0)
    assert item is not None
    assert item.data(Qt.ItemDataRole.UserRole) == orphan_id

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_refresh_runs_used_detection_not_in_saved_list(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    """DetectionSettings referenced by a CaliResult must NOT appear in saved-segs."""
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    assert runs_panel._runs_list.count() == 1
    assert runs_panel._saved_segs_list.count() == 0

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


# ============================================================================
# Signals
# ============================================================================


def test_segmentation_selected_signal_emitted(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        orphan = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(orphan)
        session.commit()
        orphan_id = orphan.id
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    received: list[int] = []
    runs_panel.segmentationSelected.connect(received.append)

    item = runs_panel._saved_segs_list.item(0)
    assert item is not None
    runs_panel._on_saved_seg_clicked(item)

    assert received == [orphan_id]

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_run_selected_signal_emitted(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        _, run_id = _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    received: list[int] = []
    runs_panel.runSelected.connect(received.append)

    item = runs_panel._runs_list.item(0)
    assert item is not None
    runs_panel._on_run_item_clicked(item)

    assert received == [run_id]

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


# ============================================================================
# Mutual exclusion of list selections
# ============================================================================


def _make_db_with_run_and_orphan(tmp_path: Path) -> Path:
    """DB with one run (detection used) and one orphan detection."""
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        _add_detection_and_run(session)
        orphan = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(orphan)
        session.commit()
    engine.dispose(close=True)
    return db_path


def test_selecting_run_clears_saved_segs_selection(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    db_path = _make_db_with_run_and_orphan(tmp_path)
    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    # Pre-select the saved seg
    seg_item = runs_panel._saved_segs_list.item(0)
    runs_panel._saved_segs_list.setCurrentItem(seg_item)
    qtbot.wait(20)
    assert runs_panel._saved_segs_list.selectedItems()

    # Select the run
    run_item = runs_panel._runs_list.item(0)
    runs_panel._runs_list.setCurrentItem(run_item)
    qtbot.wait(20)

    assert not runs_panel._saved_segs_list.selectedItems()

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_selecting_saved_seg_clears_run_selection(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    db_path = _make_db_with_run_and_orphan(tmp_path)
    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    # Pre-select the run
    run_item = runs_panel._runs_list.item(0)
    runs_panel._runs_list.setCurrentItem(run_item)
    qtbot.wait(20)
    assert runs_panel._runs_list.selectedItems()

    # Select the saved seg
    seg_item = runs_panel._saved_segs_list.item(0)
    runs_panel._saved_segs_list.setCurrentItem(seg_item)
    qtbot.wait(20)

    assert not runs_panel._runs_list.selectedItems()

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


# ============================================================================
# Delete button state
# ============================================================================


@pytest.mark.parametrize(
    "select_run,select_seg,expect_enabled",
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_delete_button_state(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    select_run: bool,
    select_seg: bool,
    expect_enabled: bool,
) -> None:
    db_path = _make_db_with_run_and_orphan(tmp_path)
    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._runs_list.clearSelection()
    runs_panel._saved_segs_list.clearSelection()
    runs_panel._delete_btn.setEnabled(False)

    if select_run:
        runs_panel._runs_list.setCurrentItem(runs_panel._runs_list.item(0))
    if select_seg:
        runs_panel._saved_segs_list.setCurrentItem(runs_panel._saved_segs_list.item(0))

    runs_panel._update_delete_button()
    assert runs_panel._delete_btn.isEnabled() is expect_enabled

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


# ============================================================================
# get_selected_saved_segmentation_id() / get_selected_detection_settings_id()
# ============================================================================


def test_get_selected_saved_segmentation_id_none_when_nothing_selected(
    runs_panel: _RunsPanel,
) -> None:
    assert runs_panel.get_selected_saved_segmentation_id() is None


def test_get_selected_saved_segmentation_id_returns_id(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        orphan = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(orphan)
        session.commit()
        orphan_id = orphan.id
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._saved_segs_list.setCurrentItem(runs_panel._saved_segs_list.item(0))

    assert runs_panel.get_selected_saved_segmentation_id() == orphan_id

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_get_selected_detection_settings_id_none_when_nothing_selected(
    runs_panel: _RunsPanel,
) -> None:
    assert runs_panel.get_selected_detection_settings_id() is None


def test_get_selected_detection_settings_id_from_saved_seg(
    tmp_path: Path, runs_panel: _RunsPanel, qtbot: QtBot
) -> None:
    """When no run is selected, returns the saved seg's DetectionSettings ID."""
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        orphan = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(orphan)
        session.commit()
        orphan_id = orphan.id
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._runs_list.clearSelection()
    runs_panel._saved_segs_list.setCurrentItem(runs_panel._saved_segs_list.item(0))

    assert runs_panel.get_selected_detection_settings_id() == orphan_id

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


# ============================================================================
# Internal DB helpers
# ============================================================================


def test_count_runs_using_detection_no_database(runs_panel: _RunsPanel) -> None:
    assert runs_panel._count_runs_using_detection(1) == 0


def test_count_runs_using_detection(tmp_path: Path, runs_panel: _RunsPanel) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        d = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d)
        session.flush()
        for _ in range(2):
            session.add(
                CaliResult(
                    experiment=1,
                    detection_settings_id=d.id,
                    positions_detected=[0],
                )
            )
        session.commit()
        detection_id = d.id
    engine.dispose(close=True)

    runs_panel._database_path = db_path
    assert runs_panel._count_runs_using_detection(detection_id) == 2


def test_get_detection_id_for_run_no_database(runs_panel: _RunsPanel) -> None:
    assert runs_panel._get_detection_id_for_run(999) is None


def test_get_detection_id_for_run(tmp_path: Path, runs_panel: _RunsPanel) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        detection_id, run_id = _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel._database_path = db_path
    assert runs_panel._get_detection_id_for_run(run_id) == detection_id


# ============================================================================
# _delete_run_from_database() — keep_detection flag
# ============================================================================


def test_delete_run_keep_detection_preserves_detection(
    tmp_path: Path, runs_panel: _RunsPanel
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        detection_id, run_id = _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel._database_path = db_path
    runs_panel._delete_run_from_database(run_id, keep_detection=True)

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(CaliResult, run_id) is None
        assert session.get(DetectionSettings, detection_id) is not None
    engine2.dispose(close=True)


def test_delete_run_without_keep_detection_removes_detection(
    tmp_path: Path, runs_panel: _RunsPanel
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        detection_id, run_id = _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel._database_path = db_path
    runs_panel._delete_run_from_database(run_id, keep_detection=False)

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(CaliResult, run_id) is None
        assert session.get(DetectionSettings, detection_id) is None
    engine2.dispose(close=True)


# ============================================================================
# _delete_detection_data()
# ============================================================================


def test_delete_detection_data_removes_detection_rois_and_empty_fov(
    tmp_path: Path, runs_panel: _RunsPanel
) -> None:
    from sqlmodel import Session, create_engine

    from cali.sqlmodel._model import FOV, ROI

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        fov = FOV(name="A1_0000", position_index=0)
        session.add(fov)
        session.flush()
        d = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d)
        session.flush()
        roi = ROI(fov_id=fov.id, label_value=1, detection_settings_id=d.id)
        session.add(roi)
        session.commit()
        detection_id = d.id
        fov_id = fov.id
    engine.dispose(close=True)

    runs_panel._database_path = db_path
    runs_panel._delete_detection_data(detection_id)

    from sqlmodel import Session, create_engine, select

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(DetectionSettings, detection_id) is None
        assert (
            session.exec(
                select(ROI).where(ROI.detection_settings_id == detection_id)
            ).all()
            == []
        )
        # FOV becomes empty → should be removed
        assert session.get(FOV, fov_id) is None
    engine2.dispose(close=True)


# ============================================================================
# _clear_all_from_database()
# ============================================================================


def test_clear_all_from_database_keeps_specified_detection(
    tmp_path: Path, runs_panel: _RunsPanel
) -> None:
    from sqlmodel import Session, create_engine, select

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        keep_id, _ = _add_detection_and_run(session, model_type="cpsam")
        delete_id, _ = _add_detection_and_run(session, model_type="cyto3")
        session.commit()
    engine.dispose(close=True)

    runs_panel._database_path = db_path
    runs_panel._clear_all_from_database(keep_detection_ids={keep_id})

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(DetectionSettings, keep_id) is not None
        assert session.get(DetectionSettings, delete_id) is None
        assert session.exec(select(CaliResult)).all() == []
    engine2.dispose(close=True)


def test_clear_all_from_database_deletes_everything_by_default(
    tmp_path: Path, runs_panel: _RunsPanel
) -> None:
    from sqlmodel import Session, create_engine, select

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        detection_id, _ = _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel._database_path = db_path
    runs_panel._clear_all_from_database()

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(DetectionSettings, detection_id) is None
        assert session.exec(select(CaliResult)).all() == []
    engine2.dispose(close=True)


# ============================================================================
# _delete_selected_saved_segmentation()
# ============================================================================


def test_delete_selected_saved_segmentation_confirmed(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        orphan = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(orphan)
        session.commit()
        orphan_id = orphan.id
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._saved_segs_list.setCurrentItem(runs_panel._saved_segs_list.item(0))
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **kw: QMessageBox.StandardButton.Yes
    )

    emitted: list[bool] = []
    runs_panel.settingsDeleted.connect(lambda: emitted.append(True))
    runs_panel._delete_selected_saved_segmentation()
    qtbot.wait(50)

    assert runs_panel._saved_segs_list.count() == 0
    assert emitted

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(DetectionSettings, orphan_id) is None
    engine2.dispose(close=True)

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_delete_selected_saved_segmentation_cancelled(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        orphan = DetectionSettings(method="cellpose", model_type="cyto3")
        session.add(orphan)
        session.commit()
        orphan_id = orphan.id
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._saved_segs_list.setCurrentItem(runs_panel._saved_segs_list.item(0))
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **kw: QMessageBox.StandardButton.No
    )

    runs_panel._delete_selected_saved_segmentation()
    qtbot.wait(50)

    assert runs_panel._saved_segs_list.count() == 1

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(DetectionSettings, orphan_id) is not None
    engine2.dispose(close=True)

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


# ============================================================================
# _delete_selected_run() — sole detection keep/delete/cancel
# ============================================================================


@pytest.mark.parametrize(
    "keep_choice,detection_survives",
    [(True, True), (False, False)],
)
def test_delete_selected_run_sole_detection(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    keep_choice: bool,
    detection_survives: bool,
) -> None:
    """When run is sole user of detection, keep_choice determines if detection survives."""  # noqa: E501
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        detection_id, run_id = _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._runs_list.setCurrentItem(runs_panel._runs_list.item(0))
    monkeypatch.setattr(
        runs_panel, "_ask_keep_or_delete_segmentation", lambda *a: keep_choice
    )

    runs_panel._delete_selected_run()
    qtbot.wait(50)

    assert runs_panel._runs_list.count() == 0

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(CaliResult, run_id) is None
        det = session.get(DetectionSettings, detection_id)
        assert (det is not None) is detection_survives
    engine2.dispose(close=True)

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_delete_selected_run_sole_detection_cancel(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        _, _run_id = _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._runs_list.setCurrentItem(runs_panel._runs_list.item(0))
    monkeypatch.setattr(runs_panel, "_ask_keep_or_delete_segmentation", lambda *a: None)

    runs_panel._delete_selected_run()
    qtbot.wait(50)

    # Nothing deleted after cancel
    assert runs_panel._runs_list.count() == 1

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_delete_selected_run_shared_detection_shows_simple_confirm(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared detection → simple Yes/No QMessageBox (no keep dialog)."""
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        d = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d)
        session.flush()
        r1 = CaliResult(
            experiment=1, detection_settings_id=d.id, positions_detected=[0]
        )
        r2 = CaliResult(
            experiment=1, detection_settings_id=d.id, positions_detected=[1]
        )
        session.add_all([r1, r2])
        session.commit()
        detection_id = d.id
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    runs_panel._runs_list.setCurrentItem(runs_panel._runs_list.item(0))
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **kw: QMessageBox.StandardButton.Yes
    )

    runs_panel._delete_selected_run()
    qtbot.wait(50)

    # One run deleted, other remains; shared detection must still exist
    assert runs_panel._runs_list.count() == 1
    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(DetectionSettings, detection_id) is not None
    engine2.dispose(close=True)

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


# ============================================================================
# _delete_selected() routing
# ============================================================================


def test_delete_selected_routes_to_saved_seg_handler(
    runs_panel: _RunsPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        runs_panel, "_delete_selected_saved_segmentation", lambda: called.append("seg")
    )
    monkeypatch.setattr(
        runs_panel, "_delete_selected_run", lambda: called.append("run")
    )

    item = QListWidgetItem("test-seg")
    runs_panel._saved_segs_list.addItem(item)
    runs_panel._saved_segs_list.setCurrentItem(item)

    runs_panel._delete_selected()
    assert called == ["seg"]


def test_delete_selected_routes_to_run_handler(
    runs_panel: _RunsPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        runs_panel, "_delete_selected_saved_segmentation", lambda: called.append("seg")
    )
    monkeypatch.setattr(
        runs_panel, "_delete_selected_run", lambda: called.append("run")
    )

    # No saved-seg selected → goes to run handler
    runs_panel._delete_selected()
    assert called == ["run"]


# ============================================================================
# _build_summary() / _all_detection_summaries()
# ============================================================================


def test_build_summary_counts(tmp_path: Path, runs_panel: _RunsPanel) -> None:
    from sqlmodel import Session, create_engine

    from cali.sqlmodel._model import FOV, ROI

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        fov1 = FOV(name="A1_0000", position_index=0)
        fov2 = FOV(name="A1_0001", position_index=1)
        session.add_all([fov1, fov2])
        session.flush()

        d = DetectionSettings(method="cellpose", model_type="cpsam")
        session.add(d)
        session.flush()

        session.add(
            CaliResult(experiment=1, detection_settings_id=d.id, positions_detected=[0])
        )

        for fov_id, n_rois in [(fov1.id, 2), (fov2.id, 1)]:
            for label in range(1, n_rois + 1):
                session.add(
                    ROI(fov_id=fov_id, label_value=label, detection_settings_id=d.id)
                )

        session.commit()
        d_settings = session.get(DetectionSettings, d.id)
        summary = runs_panel._build_summary(session, d_settings)

    engine.dispose(close=True)

    assert summary.detection_id == d.id
    assert summary.run_count == 1
    assert summary.roi_count == 3
    assert summary.fov_count == 2


def test_all_detection_summaries_no_database(runs_panel: _RunsPanel) -> None:
    assert runs_panel._all_detection_summaries() == []


# ============================================================================
# _clear_all_runs() — dialog interactions
# ============================================================================


def test_clear_all_runs_cancelled_does_nothing(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        _add_detection_and_run(session)
        session.commit()
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    monkeypatch.setattr(
        _DetectionKeepDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )

    runs_panel._clear_all_runs()
    qtbot.wait(50)

    assert runs_panel._runs_list.count() == 1

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()


def test_clear_all_runs_accepted_with_kept_detection(
    tmp_path: Path,
    runs_panel: _RunsPanel,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlmodel import Session, create_engine

    db_path = _make_db(tmp_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        keep_id, _ = _add_detection_and_run(session, model_type="cpsam")
        delete_id, _ = _add_detection_and_run(session, model_type="cyto3")
        session.commit()
    engine.dispose(close=True)

    runs_panel.set_database_path(db_path)
    qtbot.wait(50)

    monkeypatch.setattr(
        _DetectionKeepDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(
        _DetectionKeepDialog, "kept_detection_ids", lambda self: {keep_id}
    )

    runs_panel._clear_all_runs()
    qtbot.wait(50)

    # All runs gone; kept detection becomes a saved seg
    assert runs_panel._runs_list.count() == 0
    assert runs_panel._saved_segs_list.count() == 1

    engine2 = create_engine(f"sqlite:///{db_path}")
    with Session(engine2) as session:
        assert session.get(DetectionSettings, keep_id) is not None
        assert session.get(DetectionSettings, delete_id) is None
    engine2.dispose(close=True)

    runs_panel.clear()
    runs_panel._database_path = None
    qtbot.wait(50)
    gc.collect()
