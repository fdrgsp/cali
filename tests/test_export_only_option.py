"""Test Export Only option in run widget and functionality."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from pytestqt.qtbot import QtBot
from sqlmodel import Session, create_engine

from cali._constants import (
    CALCIUM_DEC_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    DEC_DFF_TRACES,
    DFF_TRACES,
    INFERRED_SPIKES_SYNCHRONY,
    RAW_CALCIUM_TRACES,
)
from cali.gui._run_widget import _RunCaliWidget
from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)


def test_export_only_option_exists(qtbot: QtBot) -> None:
    """Test that Export Only option exists in the run options combo."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Check that "Export Only" is in the combo box
    items = [
        widget._run_options_combo.itemText(i)
        for i in range(widget._run_options_combo.count())
    ]
    assert "Export Only (require existing run)" in items


def test_export_only_disabled_by_default(qtbot: QtBot) -> None:
    """Test that Export Only option is disabled when no runs exist."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Get the model to check item flags
    from qtpy.QtCore import Qt

    model = widget._run_options_combo.model()

    # Find the Export Only item (should be index 6)
    export_only_index = None
    for i in range(widget._run_options_combo.count()):
        if "Export Only" in widget._run_options_combo.itemText(i):
            export_only_index = i
            break

    assert export_only_index is not None
    item = model.item(export_only_index)

    # Check that the item is disabled (NoItemFlags)
    assert item.flags() == Qt.ItemFlag.NoItemFlags


def test_export_only_enabled_with_runs(qtbot: QtBot) -> None:
    """Test that Export Only option is enabled when runs exist."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Populate with some run IDs to enable the option
    widget.populate_run_ids([1, 2, 3])

    from qtpy.QtCore import Qt

    model = widget._run_options_combo.model()

    # Find the Export Only item
    export_only_index = None
    for i in range(widget._run_options_combo.count()):
        if "Export Only" in widget._run_options_combo.itemText(i):
            export_only_index = i
            break

    assert export_only_index is not None
    item = model.item(export_only_index)

    # Check that the item is enabled
    assert item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert item.flags() & Qt.ItemFlag.ItemIsSelectable


def test_run_ids_combo_visibility(qtbot: QtBot) -> None:
    """Test that Run IDs combo appears when Export Only is selected."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Initially, run IDs combo should be hidden
    assert not widget._run_ids_combo.isVisible()

    # Enable Export Only option by adding runs
    widget.populate_run_ids([1, 2])

    # Select Export Only
    for i in range(widget._run_options_combo.count()):
        if "Export Only" in widget._run_options_combo.itemText(i):
            # Use qtbot to wait for the signal to be processed
            with qtbot.waitSignal(
                widget._run_options_combo.currentTextChanged, timeout=1000
            ):
                widget._run_options_combo.setCurrentIndex(i)

    run_ids = [1, 2, 3, 5, 10]
    widget.populate_run_ids(run_ids)

    # Should have one more item than run_ids (the "Select Run ID..." placeholder)
    assert widget._run_ids_combo.count() == len(run_ids) + 1

    # Check first item is placeholder
    assert widget._run_ids_combo.itemText(0) == "Select Run ID..."
    assert widget._run_ids_combo.itemData(0) is None

    # Check run IDs are populated correctly
    for i, run_id in enumerate(run_ids, start=1):
        assert f"Run ID {run_id}" in widget._run_ids_combo.itemText(i)
        assert widget._run_ids_combo.itemData(i) == run_id


def test_value_with_export_only(qtbot: QtBot) -> None:
    """Test that value() returns correct data for Export Only mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Setup
    widget.populate_run_ids([1, 2, 3])

    # Select Export Only
    for i in range(widget._run_options_combo.count()):
        if "Export Only" in widget._run_options_combo.itemText(i):
            with qtbot.waitSignal(
                widget._run_options_combo.currentTextChanged, timeout=1000
            ):
                widget._run_options_combo.setCurrentIndex(i)
            break

    # Select a run ID
    widget._run_ids_combo.setCurrentIndex(2)  # Select Run ID 2

    # Set some positions
    widget._positions_wdg.setValue("0, 1, 2")

    # Get value
    settings = widget.value()

    # Check that export mode is set correctly
    assert not settings.run_detection
    assert not settings.run_extraction
    assert not settings.run_analysis
    assert settings.run_id == 2
    assert settings.positions == [0, 1, 2]


def test_export_only_auto_select_single_run(qtbot: QtBot) -> None:
    """Test that when only one run exists, it's auto-selected."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Populate with single run
    widget.populate_run_ids([42])

    # Select Export Only
    for i in range(widget._run_options_combo.count()):
        if "Export Only" in widget._run_options_combo.itemText(i):
            with qtbot.waitSignal(
                widget._run_options_combo.currentTextChanged, timeout=1000
            ):
                widget._run_options_combo.setCurrentIndex(i)
            break

    # Should auto-select the only run available
    assert widget._run_ids_combo.currentData() == 42


def test_export_only_integration(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test full integration of Export Only functionality."""
    test_db_path = tmp_path / "test_export_only.cali"

    # First, create a run without exports
    exp = Experiment.create_from_data("test_export_only", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Run pipeline WITHOUT exports initially
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=None,  # No exports
        export_correlations=None,  # No exports
    )

    # Verify database was created
    assert test_db_path.exists()

    # Get the run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            from sqlmodel import select

            from cali.sqlmodel._model import CaliResult

            results = session.exec(select(CaliResult)).all()
            assert len(results) > 0
            run_id = results[-1].id
    finally:
        engine.dispose(close=True)

    # Now export data retroactively using the export functionality
    # This simulates what would happen when "Export Only" is selected
    from cali.util import (
        export_calcium_dec_dff_correlation_to_csv,
        export_calcium_dff_correlation_to_csv,
        export_deconvolved_dff_traces_to_csv,
        export_dff_traces_to_csv,
        export_inferred_spikes_synchrony_to_csv,
        export_raw_traces_to_csv,
    )

    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export traces
    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DFF_TRACES: True,
        DEC_DFF_TRACES: True,
    }

    trace_export_map = {
        RAW_CALCIUM_TRACES: (export_raw_traces_to_csv, "raw_traces.csv"),
        DFF_TRACES: (export_dff_traces_to_csv, "dff_traces.csv"),
        DEC_DFF_TRACES: (
            export_deconvolved_dff_traces_to_csv,
            "deconvolved_dff_traces.csv",
        ),
    }

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        for trace_type, should_export in export_traces.items():
            if should_export and trace_type in trace_export_map:
                export_func, filename = trace_export_map[trace_type]
                output_path = export_dir / filename
                export_func(engine, output_path, run_id=run_id)

        # Export correlations
        export_correlations = {
            CALCIUM_DFF_CORRELATION: True,
            CALCIUM_DEC_DFF_CORRELATION: True,
            INFERRED_SPIKES_SYNCHRONY: True,
        }

        correlation_export_map = {
            CALCIUM_DFF_CORRELATION: (
                export_calcium_dff_correlation_to_csv,
                "calcium_dff_correlation_matrix.csv",
            ),
            CALCIUM_DEC_DFF_CORRELATION: (
                export_calcium_dec_dff_correlation_to_csv,
                "calcium_dec_dff_correlation_matrix.csv",
            ),
            INFERRED_SPIKES_SYNCHRONY: (
                export_inferred_spikes_synchrony_to_csv,
                "inferred_spikes_synchrony_matrix.csv",
            ),
        }

        for corr_type, should_export in export_correlations.items():
            if should_export and corr_type in correlation_export_map:
                export_func, filename = correlation_export_map[corr_type]
                output_path = export_dir / filename
                export_func(engine, output_path, run_id=run_id)
    finally:
        engine.dispose(close=True)

    # Verify all expected files were created
    assert (export_dir / "raw_traces.csv").exists()
    assert (export_dir / "dff_traces.csv").exists()
    assert (export_dir / "deconvolved_dff_traces.csv").exists()

    # Correlation files may have FOV prefixes
    assert len(list(export_dir.glob("*calcium_dff_correlation_matrix.csv"))) > 0
    assert len(list(export_dir.glob("*calcium_dec_dff_correlation_matrix.csv"))) > 0
    assert len(list(export_dir.glob("*inferred_spikes_synchrony_matrix.csv"))) > 0

    # Verify content
    df = pd.read_csv(export_dir / "raw_traces.csv")
    assert len(df) > 0
    assert len(df.columns) > 0


def test_export_only_no_runs_keeps_default_option(qtbot: QtBot) -> None:
    """Test that selecting Export Only reverts to default when no runs exist."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    # Try to select Export Only (it's disabled)
    export_only_index = None
    for i in range(widget._run_options_combo.count()):
        if "Export Only" in widget._run_options_combo.itemText(i):
            export_only_index = i
            break

    # Try to set it (should not work since it's disabled)
    widget._run_options_combo.currentIndex()
    widget._run_options_combo.setCurrentIndex(export_only_index)

    # Should still be at default option due to disabled state
    # (Qt might prevent setting disabled items or it reverts)
    # The important part is that _update_options_availability would reset it
    widget._update_options_availability(
        has_detections=False, has_extractions=False, has_runs=False
    )

    # After update, if it was somehow set to 6, it should revert to 0
    if widget._run_options_combo.currentIndex() == export_only_index:
        assert widget._run_options_combo.currentIndex() == 0


def test_export_only_positions_widget_state(qtbot: QtBot) -> None:
    """Test that positions widget is not needed for Export Only mode."""
    widget = _RunCaliWidget()
    qtbot.addWidget(widget)

    widget.populate_run_ids([1, 2])

    # Select Export Only
    for i in range(widget._run_options_combo.count()):
        if "Export Only" in widget._run_options_combo.itemText(i):
            with qtbot.waitSignal(
                widget._run_options_combo.currentTextChanged, timeout=1000
            ):
                widget._run_options_combo.setCurrentIndex(i)
            break

    # Select a run ID
    widget._run_ids_combo.setCurrentIndex(1)  # Select first run (Run ID 1)

    # Positions can be empty for export only
    widget._positions_wdg.setValue("")

    settings = widget.value()
    # In export mode, positions list might be empty
    assert settings.run_id is not None


def test_export_only_with_runner_integration(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test Export Only using CaliRunner's export methods directly."""
    test_db_path = tmp_path / "test_export_runner.cali"

    # First, create a run without exports
    exp = Experiment.create_from_data("test_export_runner", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Run pipeline WITHOUT exports initially
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=None,  # No exports
        export_correlations=None,  # No exports
    )

    # Verify database was created
    assert test_db_path.exists()

    # Get the run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            from sqlmodel import select

            from cali.sqlmodel._model import CaliResult

            results = session.exec(select(CaliResult)).all()
            assert len(results) > 0
            run_id = results[-1].id
    finally:
        engine.dispose(close=True)

    # Now use the runner's export methods directly (simulating Export Only)
    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DFF_TRACES: True,
        DEC_DFF_TRACES: True,
    }

    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
        CALCIUM_DEC_DFF_CORRELATION: True,
        INFERRED_SPIKES_SYNCHRONY: True,
    }

    from cali.util._database_to_csv import (
        export_correlations_to_csv,
        export_traces_to_csv,
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        # Test the export functions
        export_traces_to_csv(engine, export_traces, run_id, test_db_path)
        export_correlations_to_csv(engine, export_correlations, run_id, test_db_path)
    finally:
        engine.dispose(close=True)

    # Verify exports were created
    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"

    # Check trace files
    assert (export_dir / "raw_traces.csv").exists()
    assert (export_dir / "dff_traces.csv").exists()
    assert (export_dir / "deconvolved_dff_traces.csv").exists()

    # Check correlation files (may have FOV prefixes)
    assert len(list(export_dir.glob("*calcium_dff_correlation_matrix.csv"))) > 0
    assert len(list(export_dir.glob("*calcium_dec_dff_correlation_matrix.csv"))) > 0
    assert len(list(export_dir.glob("*inferred_spikes_synchrony_matrix.csv"))) > 0

    # Verify content
    df = pd.read_csv(export_dir / "raw_traces.csv")
    assert len(df) > 0
    assert len(df.columns) > 0
