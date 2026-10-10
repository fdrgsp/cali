"""Explicit GUI source choices use verified comparisons and transactional repairs."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QDialog
from sqlmodel import Session, select

from cali.gui import CaliGui
from cali.gui._run_widget import ExtractionSourceOption, _RunCaliWidget
from cali.gui._runs_panel import _RunsPanel
from cali.gui._source_repair_dialog import _SourceRepairDialog
from cali.sqlmodel import (
    CaliResult,
    Experiment,
    MigrationIssue,
    create_cali_engine,
    preview_legacy_result_source,
)
from tests.test_runner_sources import _database
from tests.test_source_stage_audit import _legacy


def test_preview_verifies_without_mutation_or_flushing_pending_work(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preview.cali"
    source_id, target_id, _ = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        target = session.get(CaliResult, target_id)
        trace = target.traces[0]
        previous = trace.model_dump_json()
        old_window = trace.extraction_frame_window
        old_run = trace.get_spike_trace("oasis").inference_run
        pending = Experiment(name="unrelated pending work")
        session.add(pending)
        preview = preview_legacy_result_source(session, target_id, source_id)
        assert preview.result_id == target_id
        assert preview.source_result_id == source_id
        assert preview.trace_count == 1
        assert preview.roi_metric_count == preview.fov_metric_count == 1
        assert preview.spike_outputs == (("oasis", "a.u."),)
        assert preview.retained_frame_counts == (3,)
        assert preview.source_start_frames == (0,)
        assert trace.model_dump_json() == previous
        assert trace.extraction_frame_window is old_window
        assert trace.get_spike_trace("oasis").inference_run is old_run
        assert target.legacy_trace_resolution == "unresolved_stage_flags"
        assert pending.id is None
        assert set(session.new) == {pending}
        assert not session.dirty
        assert len(session.exec(select(MigrationIssue)).all()) == 6
        session.rollback()
    engine.dispose()


def test_preview_rejects_mismatches_without_modifying_the_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mismatch.cali"
    source_id, target_id, _ = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        source = session.get(CaliResult, source_id)
        source.traces[0].raw_trace = [8, 8, 8]
        session.commit()
        with pytest.raises(ValueError, match="does not match"):
            preview_legacy_result_source(session, target_id, source_id)
        assert not session.dirty and not session.new
        assert session.get(CaliResult, target_id).source_extraction_result_id is None
    engine.dispose()


def test_dialog_requires_an_explicit_verified_selection(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "dialog.cali"
    source_id, target_id, dependent_id = _legacy(path)
    dialog = _SourceRepairDialog(path, target_id)
    qtbot.addWidget(dialog)
    assert dialog._sources_combo.count() == 2
    assert dialog._sources_combo.findData(dependent_id) == -1
    assert dialog.selected_source_id() is None
    assert not dialog._apply_btn.isEnabled()
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Rejected
    dialog._sources_combo.setCurrentIndex(dialog._sources_combo.findData(source_id))
    assert dialog.selected_source_id() == source_id
    assert dialog._apply_btn.isEnabled()
    text = dialog._comparison_label.text()
    assert "Verified match" in text and "OASIS (a.u.)" in text
    assert "Matching ROI traces: 1" in text
    dialog._sources_combo.setCurrentIndex(0)
    assert dialog.selected_source_id() is None
    assert not dialog._apply_btn.isEnabled()
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        assert session.get(CaliResult, target_id).source_extraction_result_id is None
    engine.dispose()


def test_dialog_reports_mismatch_and_does_not_enable_repair(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "dialog_mismatch.cali"
    source_id, target_id, _ = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        session.get(CaliResult, source_id).traces[0].raw_trace = [9, 9, 9]
        session.commit()
    engine.dispose()
    dialog = _SourceRepairDialog(path, target_id)
    qtbot.addWidget(dialog)
    dialog._sources_combo.setCurrentIndex(dialog._sources_combo.findData(source_id))
    assert "Run a fresh analysis" in dialog._comparison_label.text()
    assert dialog.selected_source_id() is None
    assert not dialog._apply_btn.isEnabled()


def test_missing_database_does_not_get_created(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "missing.cali"
    dialog = _SourceRepairDialog(path, 1)
    qtbot.addWidget(dialog)
    assert not dialog._apply_btn.isEnabled()
    assert "no longer available" in dialog._comparison_label.text()
    assert not path.exists()


def test_no_resolved_sources_explains_need_for_fresh_extraction(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_source.cali"
    source_id, target_id, _ = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        session.get(CaliResult, source_id).positions_extracted = None
        session.commit()
    engine.dispose()
    dialog = _SourceRepairDialog(path, target_id)
    qtbot.addWidget(dialog)
    assert "fresh extraction and analysis" in dialog._comparison_label.text()
    assert not dialog._apply_btn.isEnabled()


def test_panel_repairs_selected_run_preserving_values_and_dependent_audits(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "panel.cali"
    source_id, target_id, dependent_id = _legacy(path)
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel.set_database_path(path)
    panel.select_run_by_id(source_id)
    assert not panel._repair_source_btn.isEnabled()
    panel.select_run_by_id(target_id)
    assert panel._repair_source_btn.isEnabled()

    def accept_verified(dialog: _SourceRepairDialog) -> int:
        dialog._sources_combo.setCurrentIndex(dialog._sources_combo.findData(source_id))
        assert dialog._apply_btn.isEnabled()
        return QDialog.DialogCode.Accepted

    with (
        patch.object(_SourceRepairDialog, "exec", accept_verified),
        qtbot.waitSignal(panel.sourceRepaired) as repaired,
    ):
        panel._repair_selected_source()
    assert repaired.args == [target_id]
    assert panel.get_selected_run_id() == target_id
    assert not panel._repair_source_btn.isEnabled()
    assert "needs review" not in panel._runs_list.currentItem().text()
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        target = session.get(CaliResult, target_id)
        assert target.source_extraction_result_id == source_id
        assert target.traces[0].raw_trace == [1, 2, 1]
        assert (
            target.data_analysis_results[0].get_spike_analysis("oasis").threshold
            == 0.25
        )
        assert (
            target.fov_analysis_results[0].get_spike_analysis("oasis").spike_burst_count
            == 2
        )
        assert session.get(CaliResult, dependent_id).source_extraction_result_id is None
        issues = session.exec(
            select(MigrationIssue).where(MigrationIssue.analysis_result_id == target_id)
        ).all()
        assert all(issue.resolved for issue in issues)
        assert any(issue.code == "legacy_source_selected" for issue in issues)
    engine.dispose()


def test_cancelling_source_dialog_leaves_quarantine_unchanged(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cancel.cali"
    _, target_id, _ = _legacy(path)
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel.set_database_path(path)
    panel.select_run_by_id(target_id)
    with patch.object(
        _SourceRepairDialog, "exec", return_value=QDialog.DialogCode.Rejected
    ):
        panel._repair_selected_source()
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        assert session.get(CaliResult, target_id).source_extraction_result_id is None
        assert not session.exec(
            select(MigrationIssue).where(
                MigrationIssue.code == "legacy_source_selected"
            )
        ).all()
    engine.dispose()


def test_source_is_revalidated_after_dialog_and_failure_rolls_back(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "changed_source.cali"
    source_id, target_id, _ = _legacy(path)
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel.set_database_path(path)
    panel.select_run_by_id(target_id)

    def accept_changed_source(dialog: _SourceRepairDialog) -> int:
        dialog._sources_combo.setCurrentIndex(dialog._sources_combo.findData(source_id))
        assert dialog._apply_btn.isEnabled()
        engine = create_cali_engine(f"sqlite:///{path}")
        with Session(engine) as session:
            session.get(CaliResult, source_id).traces[0].raw_trace = [9, 9, 9]
            session.commit()
        engine.dispose()
        return QDialog.DialogCode.Accepted

    with (
        patch.object(_SourceRepairDialog, "exec", accept_changed_source),
        patch("cali.gui._runs_panel.QMessageBox.warning") as error,
    ):
        panel._repair_selected_source()
    assert "does not match" in error.call_args.args[2]
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        target = session.get(CaliResult, target_id)
        assert target.source_extraction_result_id is None
        assert target.traces[0].raw_trace == [1, 2, 1]
        assert not session.exec(
            select(MigrationIssue).where(
                MigrationIssue.code == "legacy_source_selected"
            )
        ).all()
    engine.dispose()


def test_commit_failure_rolls_back_all_repair_links_and_audit(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "commit_failure.cali"
    source_id, target_id, _ = _legacy(path)
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel.set_database_path(path)
    panel.select_run_by_id(target_id)

    def accept_verified(dialog: _SourceRepairDialog) -> int:
        dialog._sources_combo.setCurrentIndex(dialog._sources_combo.findData(source_id))
        return QDialog.DialogCode.Accepted

    with (
        patch.object(_SourceRepairDialog, "exec", accept_verified),
        patch.object(Session, "commit", side_effect=RuntimeError("commit failed")),
        patch("cali.gui._runs_panel.QMessageBox.warning") as error,
    ):
        panel._repair_selected_source()
    assert "commit failed" in error.call_args.args[2]
    assert panel._repair_source_btn.isEnabled()
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        target = session.get(CaliResult, target_id)
        assert target.source_extraction_result_id is None
        assert target.traces[0].extraction_frame_window.extraction_result_id is None
        assert (
            target.data_analysis_results[0].get_spike_analysis("oasis").spike_trace_id
            is None
        )
        assert (
            target.fov_analysis_results[0]
            .get_spike_analysis("oasis")
            .spike_inference_run_id
            is None
        )
        assert not session.exec(
            select(MigrationIssue).where(
                MigrationIssue.code == "legacy_source_selected"
            )
        ).all()
    engine.dispose()


def test_coverage_uses_exact_source_and_source_list_omits_quarantined_owners(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "coverage.cali"
    graph = _database(path)
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel.set_database_path(path)
    assert [source.result_id for source in panel.get_extraction_sources()] == graph[-1]
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._database_path = str(path)
    assert gui._check_positions_missing_extraction(graph[1], graph[2], [0, 1, 2]) == []
    assert gui._check_positions_missing_extraction(
        graph[1], graph[2], [0, 1, 2], graph[-1][0]
    ) == [1]
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        session.get(
            CaliResult, graph[-1][0]
        ).legacy_trace_resolution = "unresolved_source"
        session.commit()
    assert [source.result_id for source in panel.get_extraction_sources()] == [
        graph[-1][1]
    ]
    engine.dispose()


def test_analysis_source_choices_filter_by_settings_and_preserve_selection(
    qtbot: QtBot,
) -> None:
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)
    widget.populate_detection_settings([(1, "cellpose"), (2, "cellpose")])
    widget.populate_extraction_settings([1, 2])
    widget.populate_extraction_sources(
        [
            ExtractionSourceOption(10, 1, 1, (0, 2)),
            ExtractionSourceOption(11, 1, 1, (1,)),
            ExtractionSourceOption(12, 2, 1, (0,)),
            ExtractionSourceOption(13, 1, 2, (0,)),
        ]
    )
    widget._run_options_combo.setCurrentIndex(5)
    widget._detection_settings_combo.setCurrentIndex(1)
    widget._extraction_settings_combo.setCurrentIndex(1)
    combo = widget._source_extraction_combo
    assert combo.count() == 3
    assert combo.findData(12) == combo.findData(13) == -1
    assert widget.value().source_extraction_result_id is None
    combo.setCurrentIndex(combo.findData(11))
    widget.populate_extraction_sources(widget._source_options)
    assert widget.value().source_extraction_result_id == 11
    widget._extraction_settings_combo.setCurrentIndex(2)
    assert combo.count() == 2 and combo.findData(13) == 1
    assert widget.value().source_extraction_result_id is None
    combo.setCurrentIndex(1)
    widget._run_options_combo.setCurrentIndex(4)
    assert widget.value().source_extraction_result_id is None
    assert combo.isHidden()
    widget.enable(False)
    assert not combo.isEnabled()
    widget.reset()
    assert combo.count() == 1


def test_gui_forwards_exact_source_and_selected_detection_to_runner(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "forward.cali"
    source_id, _, _ = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        source = session.get(CaliResult, source_id)
        detection_id, extraction_id = (
            source.detection_settings_id,
            source.extraction_settings_id,
        )
    engine.dispose()
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._database_path = str(path)
    gui._output_path = str(tmp_path)
    gui._runs_panel.set_database_path(path)
    gui._populate_settings(path)
    widget = gui._run_cali_wdg
    widget._run_options_combo.setCurrentIndex(5)
    widget._detection_settings_combo.setCurrentIndex(1)
    widget._extraction_settings_combo.setCurrentIndex(1)
    widget._source_extraction_combo.setCurrentIndex(1)
    widget._positions_wdg.setValue("0")
    gui._populate_settings(path)
    assert widget.value().source_extraction_result_id == source_id
    with (
        patch.object(gui, "_check_positions_missing_detection", return_value=[]),
        patch.object(gui, "_check_positions_missing_extraction", return_value=[]),
        patch.object(gui, "_save_plate_map_to_database"),
        patch.object(gui, "_enable"),
        patch.object(gui._detection_wdg, "to_model_settings") as detection,
        patch.object(gui._runner, "run", return_value=(item for item in ())) as run,
        patch("cali.gui._cali_gui.create_worker") as worker,
        patch("cali.gui._cali_gui.show_error_dialog") as error,
    ):
        gui._on_cali_run()
        error.assert_not_called()
        detection.assert_not_called()
        assert worker.call_count == 1
        list(worker.call_args.args[0]())
        assert run.call_args.kwargs["detection_settings"] == detection_id
        assert run.call_args.kwargs["extraction_settings"] == extraction_id
        assert run.call_args.kwargs["source_extraction_result_id"] == source_id
        worker.call_args.kwargs["_connect"]["finished"]()
    gui._elapsed_timer.stop()
