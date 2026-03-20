"""Tests for the enable_calcium / enable_spikes feature.

Covers:
- AnalysisSettings model: fields, __eq__, __hash__
- DB migration: migrate_analysis_settings
- GUI: _AnalysisGUI checkboxes, value/setValue/reset, _on_enable_changed
- FOV analysis gating: compute_fov_analysis with calcium/spikes disabled
- Settings save/load round-trip with enable flags
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sqlalchemy import text
from sqlmodel import Session, create_engine, select

from cali.sqlmodel._model import AnalysisSettings, DataAnalysis, Traces
from cali.sqlmodel._util import create_database_and_tables, migrate_analysis_settings

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine

    from cali.sqlmodel import FOV

# ---------------------------------------------------------------------------
# AnalysisSettings model: enable_calcium / enable_spikes
# ---------------------------------------------------------------------------


def test_analysis_settings_defaults() -> None:
    """New AnalysisSettings have both analyses enabled by default."""
    s = AnalysisSettings()
    assert s.enable_calcium is True
    assert s.enable_spikes is True


@pytest.mark.parametrize(
    ("ca_a", "sp_a", "ca_b", "sp_b", "equal"),
    [
        (True, True, True, True, True),
        (False, True, False, True, True),
        (True, False, True, False, True),
        (True, True, False, True, False),
        (True, True, True, False, False),
        (False, False, False, False, True),
        (False, True, True, False, False),
    ],
)
def test_analysis_settings_eq(
    ca_a: bool, sp_a: bool, ca_b: bool, sp_b: bool, equal: bool
) -> None:
    """__eq__ accounts for enable_calcium and enable_spikes."""
    a = AnalysisSettings(enable_calcium=ca_a, enable_spikes=sp_a)
    b = AnalysisSettings(enable_calcium=ca_b, enable_spikes=sp_b)
    assert (a == b) is equal


def test_analysis_settings_hash_differs_on_flags() -> None:
    """Different enable flags produce different hashes."""
    both = AnalysisSettings(enable_calcium=True, enable_spikes=True)
    calcium_only = AnalysisSettings(enable_calcium=True, enable_spikes=False)
    spikes_only = AnalysisSettings(enable_calcium=False, enable_spikes=True)
    assert hash(both) != hash(calcium_only)
    assert hash(both) != hash(spikes_only)
    assert hash(calcium_only) != hash(spikes_only)


def test_analysis_settings_hash_matches_eq() -> None:
    """Equal settings must have equal hashes."""
    a = AnalysisSettings(enable_calcium=False, enable_spikes=True)
    b = AnalysisSettings(enable_calcium=False, enable_spikes=True)
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


def test_analysis_settings_persist_enable_flags(temp_db: tuple[Engine, Path]) -> None:
    """enable_calcium / enable_spikes survive a database round-trip."""
    engine, _ = temp_db

    with Session(engine) as session:
        session.add(AnalysisSettings(enable_calcium=False, enable_spikes=True))
        session.add(AnalysisSettings(enable_calcium=True, enable_spikes=False))
        session.commit()

    with Session(engine) as session:
        rows = session.exec(
            select(AnalysisSettings).order_by(AnalysisSettings.id)
        ).all()
        assert len(rows) == 2
        assert rows[0].enable_calcium is False
        assert rows[0].enable_spikes is True
        assert rows[1].enable_calcium is True
        assert rows[1].enable_spikes is False


# ---------------------------------------------------------------------------
# DB migration
# ---------------------------------------------------------------------------


def test_migrate_adds_columns_to_legacy_db() -> None:
    """migrate_analysis_settings adds missing columns to a legacy table."""
    engine = create_engine("sqlite:///:memory:")

    # Create a "legacy" table without enable_calcium / enable_spikes
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE analysis_settings ("
                "  id INTEGER PRIMARY KEY,"
                "  peaks_height_value REAL DEFAULT 2.0"
                ")"
            )
        )
        conn.execute(
            text("INSERT INTO analysis_settings (peaks_height_value) VALUES (3.0)")
        )
        conn.commit()

    # Run migration
    migrate_analysis_settings(engine)

    # Verify columns exist with correct defaults
    with engine.connect() as conn:
        cols = {
            row[1] for row in conn.execute(text("PRAGMA table_info(analysis_settings)"))
        }
        assert "enable_calcium" in cols
        assert "enable_spikes" in cols

        row = conn.execute(
            text("SELECT enable_calcium, enable_spikes FROM analysis_settings")
        ).fetchone()
        assert row[0] == 1  # DEFAULT 1
        assert row[1] == 1

    engine.dispose(close=True)
    gc.collect()


def test_migrate_is_idempotent() -> None:
    """Calling migrate_analysis_settings twice does not raise."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    # Second call should be a no-op
    migrate_analysis_settings(engine)
    migrate_analysis_settings(engine)

    with engine.connect() as conn:
        cols = {
            row[1] for row in conn.execute(text("PRAGMA table_info(analysis_settings)"))
        }
        assert "enable_calcium" in cols
        assert "enable_spikes" in cols

    engine.dispose(close=True)
    gc.collect()


def test_migrate_noop_when_no_table() -> None:
    """migrate_analysis_settings is a no-op when the table doesn't exist."""
    engine = create_engine("sqlite:///:memory:")

    # Should not raise even though the table doesn't exist
    migrate_analysis_settings(engine)

    engine.dispose(close=True)
    gc.collect()


# ---------------------------------------------------------------------------
# GUI: _AnalysisGUI checkboxes
# ---------------------------------------------------------------------------


def test_gui_defaults(qtbot: QtBot) -> None:
    """Both checkboxes are checked by default and widgets are enabled."""
    from cali.gui._analysis_gui import _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    val = widget.value()
    assert val.enable_calcium is True
    assert val.enable_spikes is True
    assert widget._calcium_peaks_wdg.isEnabled()
    assert widget._spike_wdg.isEnabled()
    # Widget is not explicitly hidden (isVisible() requires a shown parent)
    assert not widget._n_processes_wdg.isHidden()


def test_gui_set_value_calcium_only(qtbot: QtBot) -> None:
    """setValue with enable_spikes=False disables spike widgets."""
    from cali.gui._analysis_gui import AnalysisSettingsData, _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    widget.setValue(AnalysisSettingsData(enable_calcium=True, enable_spikes=False))

    assert widget._enable_calcium_cb.isChecked()
    assert not widget._enable_spikes_cb.isChecked()
    assert widget._calcium_peaks_wdg.isEnabled()
    assert not widget._spike_wdg.isEnabled()
    assert widget._n_processes_wdg.isHidden()


def test_gui_set_value_spikes_only(qtbot: QtBot) -> None:
    """setValue with enable_calcium=False disables calcium widgets."""
    from cali.gui._analysis_gui import AnalysisSettingsData, _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    widget.setValue(AnalysisSettingsData(enable_calcium=False, enable_spikes=True))

    assert not widget._enable_calcium_cb.isChecked()
    assert widget._enable_spikes_cb.isChecked()
    assert not widget._calcium_peaks_wdg.isEnabled()
    assert widget._spike_wdg.isEnabled()
    assert not widget._n_processes_wdg.isHidden()


def test_gui_reset_restores_both(qtbot: QtBot) -> None:
    """reset() re-enables both analyses."""
    from cali.gui._analysis_gui import AnalysisSettingsData, _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Disable one
    widget.setValue(AnalysisSettingsData(enable_calcium=False, enable_spikes=True))
    assert not widget._enable_calcium_cb.isChecked()

    widget.reset()

    assert widget._enable_calcium_cb.isChecked()
    assert widget._enable_spikes_cb.isChecked()
    assert widget._calcium_peaks_wdg.isEnabled()
    assert widget._spike_wdg.isEnabled()
    assert not widget._n_processes_wdg.isHidden()


def test_gui_at_least_one_must_be_checked(qtbot: QtBot) -> None:
    """Clicking the last remaining checkbox keeps it checked (at-least-one rule)."""
    from cali.gui._analysis_gui import AnalysisSettingsData, _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Disable calcium first via setValue (doesn't fire clicked)
    widget.setValue(AnalysisSettingsData(enable_calcium=False, enable_spikes=True))
    assert not widget._enable_calcium_cb.isChecked()
    assert widget._enable_spikes_cb.isChecked()

    # Click the last remaining checked box — should bounce back
    widget._enable_spikes_cb.click()

    # At least one must remain checked
    assert widget._enable_spikes_cb.isChecked()
    val = widget.value()
    assert val.enable_spikes is True
    assert widget._spike_wdg.isEnabled()


@pytest.mark.parametrize(
    (
        "initial_calcium",
        "initial_spikes",
        "click_cb",
        "expect_calcium",
        "expect_spikes",
    ),
    [
        # Both on → uncheck calcium → calcium off, spikes on
        (True, True, "calcium", False, True),
        # Both on → uncheck spikes → calcium on, spikes off
        (True, True, "spikes", True, False),
        # Only calcium on → click calcium (only one left) → stays checked
        (True, False, "calcium", True, False),
        # Only spikes on → click spikes (only one left) → stays checked
        (False, True, "spikes", False, True),
    ],
)
def test_gui_click_enable_checkbox(
    qtbot: QtBot,
    initial_calcium: bool,
    initial_spikes: bool,
    click_cb: str,
    expect_calcium: bool,
    expect_spikes: bool,
) -> None:
    """Clicking enable checkboxes enforces at-least-one rule and updates widgets."""
    from cali.gui._analysis_gui import AnalysisSettingsData, _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    widget.setValue(
        AnalysisSettingsData(
            enable_calcium=initial_calcium, enable_spikes=initial_spikes
        )
    )

    cb = (
        widget._enable_calcium_cb if click_cb == "calcium" else widget._enable_spikes_cb
    )
    cb.click()

    assert widget._enable_calcium_cb.isChecked() is expect_calcium
    assert widget._enable_spikes_cb.isChecked() is expect_spikes
    assert widget._calcium_peaks_wdg.isEnabled() is expect_calcium
    assert widget._spike_wdg.isEnabled() is expect_spikes
    assert widget._n_processes_wdg.isHidden() is not expect_spikes


def test_gui_enable_disable_scenario(qtbot: QtBot) -> None:
    """Reproduces the reported bug: calcium-only → enable spikes → disable calcium.

    Steps:
    1. Start with only calcium checked (spikes disabled).
    2. Click calcium → should stay checked (at-least-one rule).
    3. Click spikes → spikes becomes enabled.
    4. Click calcium → calcium is now unchecked; calcium widget disabled.
    """
    from cali.gui._analysis_gui import AnalysisSettingsData, _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    # Step 1: only calcium
    widget.setValue(AnalysisSettingsData(enable_calcium=True, enable_spikes=False))
    assert widget._calcium_peaks_wdg.isEnabled()
    assert not widget._spike_wdg.isEnabled()

    # Step 2: click calcium — must bounce back
    widget._enable_calcium_cb.click()
    assert widget._enable_calcium_cb.isChecked()
    assert widget._calcium_peaks_wdg.isEnabled()

    # Step 3: enable spikes
    widget._enable_spikes_cb.click()
    assert widget._enable_calcium_cb.isChecked()
    assert widget._enable_spikes_cb.isChecked()
    assert widget._calcium_peaks_wdg.isEnabled()
    assert widget._spike_wdg.isEnabled()

    # Step 4: now uncheck calcium — should succeed and disable calcium widget
    widget._enable_calcium_cb.click()
    assert not widget._enable_calcium_cb.isChecked()
    assert widget._enable_spikes_cb.isChecked()
    assert not widget._calcium_peaks_wdg.isEnabled()
    assert widget._spike_wdg.isEnabled()


def test_gui_to_model_settings_passes_flags(qtbot: QtBot) -> None:
    """to_model_settings() propagates enable flags to AnalysisSettings."""
    from cali.gui._analysis_gui import AnalysisSettingsData, _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    widget.setValue(AnalysisSettingsData(enable_calcium=True, enable_spikes=False))
    model = widget.to_model_settings()
    assert model.enable_calcium is True
    assert model.enable_spikes is False


def test_gui_value_round_trip(qtbot: QtBot) -> None:
    """value() → setValue() → value() preserves enable flags."""
    from cali.gui._analysis_gui import _AnalysisGUI

    widget = _AnalysisGUI()
    qtbot.addWidget(widget)

    original = widget.value()
    widget.reset()
    widget.setValue(original)
    restored = widget.value()

    assert restored.enable_calcium == original.enable_calcium
    assert restored.enable_spikes == original.enable_spikes


# ---------------------------------------------------------------------------
# FOV analysis gating
# ---------------------------------------------------------------------------


def _make_fov_with_2_rois() -> FOV:
    """Helper: FOV with 2 active ROIs having spike and calcium data."""
    from cali.sqlmodel import FOV, ROI

    fov = FOV(name="test_fov", position_index=0)

    spike_pattern1 = np.zeros(50)
    spike_pattern1[[5, 15, 25, 35, 45]] = 2.0

    spike_pattern2 = np.zeros(50)
    spike_pattern2[[6, 16, 26, 36, 46]] = 2.0

    for i, spike_pattern in enumerate([spike_pattern1, spike_pattern2], start=1):
        roi = ROI(label_value=i, active=True, fov_id=fov.id)
        roi._new_traces = [
            Traces(
                dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
                den_dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
                inferred_spikes=spike_pattern.tolist(),
            )
        ]
        roi._new_data_analysis = [
            DataAnalysis(
                peaks_den_dff=[5 + (i - 1), 15 + (i - 1), 25 + (i - 1)],
                inferred_spikes_threshold=1.0,
            )
        ]
        fov.rois = [*fov.rois, roi] if fov.rois else [roi]

    return fov


def test_fov_analysis_calcium_disabled_skips_calcium() -> None:
    """When enable_calcium=False, calcium correlations and bursts are None."""
    from cali.analysis._fov_analysis import compute_fov_analysis

    fov = _make_fov_with_2_rois()
    settings = AnalysisSettings(
        enable_calcium=False,
        enable_spikes=True,
        spikes_sync_cross_corr_lag=500,
    )

    result = compute_fov_analysis(fov, settings)
    assert result is not None

    # Calcium metrics should be None
    assert result.calcium_dff_correlation_matrix is None
    assert result.calcium_den_dff_corr_matrix is None
    assert result.global_calcium_dff_correlation is None
    assert result.global_calcium_den_dff_correlation is None
    assert result.calcium_burst_count is None

    # Spike metrics should still be computed
    assert result.spike_max_lag_correlation_matrix is not None


def test_fov_analysis_spikes_disabled_skips_spikes() -> None:
    """When enable_spikes=False, spike correlations and bursts are None."""
    from cali.analysis._fov_analysis import compute_fov_analysis

    fov = _make_fov_with_2_rois()
    settings = AnalysisSettings(
        enable_calcium=True,
        enable_spikes=False,
        spikes_sync_cross_corr_lag=500,
    )

    result = compute_fov_analysis(fov, settings)
    assert result is not None

    # Spike metrics should be None
    assert result.spike_max_lag_correlation_matrix is None
    assert result.spike_jitter_synchrony_matrix is None
    assert result.global_spike_jitter_synchrony is None
    assert result.global_spike_max_lag_correlation is None
    assert result.spike_burst_count is None

    # Calcium metrics should still be computed
    assert result.calcium_dff_correlation_matrix is not None


def test_fov_analysis_both_enabled() -> None:
    """When both are enabled, all metrics are computed."""
    from cali.analysis._fov_analysis import compute_fov_analysis

    fov = _make_fov_with_2_rois()
    settings = AnalysisSettings(
        enable_calcium=True,
        enable_spikes=True,
        spikes_sync_cross_corr_lag=500,
    )

    result = compute_fov_analysis(fov, settings)
    assert result is not None

    # Both should be computed
    assert result.calcium_dff_correlation_matrix is not None
    assert result.spike_max_lag_correlation_matrix is not None


# ---------------------------------------------------------------------------
# Settings save/load with enable flags
# ---------------------------------------------------------------------------


def test_settings_json_includes_enable_flags(tmp_path: Path) -> None:
    """JSON save/load preserves enable_calcium and enable_spikes."""
    import json
    from dataclasses import asdict

    from cali.gui._analysis_gui import AnalysisSettingsData

    original = AnalysisSettingsData(enable_calcium=False, enable_spikes=True)
    data = asdict(original)

    json_path = tmp_path / "settings.json"
    json_path.write_text(json.dumps({"analysis": data}))

    loaded = json.loads(json_path.read_text())["analysis"]
    assert loaded["enable_calcium"] is False
    assert loaded["enable_spikes"] is True


def test_load_settings_missing_enable_flags_defaults_true(tmp_path: Path) -> None:
    """Loading legacy settings without enable flags defaults to True."""
    import json

    from cali.gui._analysis_gui import AnalysisSettingsData, CalciumPeaksData

    # Simulate a legacy settings file without enable flags
    legacy = {
        "calcium_peaks_data": {"peaks_height": 2.0},
    }
    json_path = tmp_path / "settings.json"
    json_path.write_text(json.dumps(legacy))

    loaded = json.loads(json_path.read_text())

    # Replicate the loading pattern from _cali_gui.py
    result = AnalysisSettingsData(
        enable_calcium=loaded.get("enable_calcium", True),
        enable_spikes=loaded.get("enable_spikes", True),
        calcium_peaks_data=(
            CalciumPeaksData(**loaded["calcium_peaks_data"])
            if loaded.get("calcium_peaks_data")
            else None
        ),
    )

    assert result.enable_calcium is True
    assert result.enable_spikes is True


def test_save_load_roundtrip_via_gui(
    qtbot: QtBot, tmp_path: Path, test_db_copy: Path
) -> None:
    """Save/load settings round-trip preserves enable flags via CaliGui."""
    import json
    from unittest.mock import patch

    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)

    settings_file = tmp_path / "settings.json"

    # Disable spikes
    from cali.gui._analysis_gui import AnalysisSettingsData

    gui._analysis_wdg.setValue(
        AnalysisSettingsData(enable_calcium=True, enable_spikes=False)
    )

    # Save
    with patch(
        "cali.gui._cali_gui.QFileDialog.getSaveFileName",
        return_value=(str(settings_file), ""),
    ):
        gui._on_save_settings()

    assert settings_file.exists()
    saved = json.loads(settings_file.read_text())
    assert saved["analysis"]["enable_calcium"] is True
    assert saved["analysis"]["enable_spikes"] is False

    # Reset and load back
    gui._analysis_wdg.reset()
    assert gui._analysis_wdg.value().enable_spikes is True

    with patch(
        "cali.gui._cali_gui.QFileDialog.getOpenFileName",
        return_value=(str(settings_file), ""),
    ):
        gui._on_load_settings()

    loaded_val = gui._analysis_wdg.value()
    assert loaded_val.enable_calcium is True
    assert loaded_val.enable_spikes is False

    gui.close()


# ---------------------------------------------------------------------------
# Run reload
# ---------------------------------------------------------------------------


def test_run_reload_restores_enable_flags(qtbot: QtBot, test_db_copy: Path) -> None:
    """Selecting a run restores enable_calcium / enable_spikes from DB."""
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)

    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
    gui._initialize_from_database(str(test_db_copy), data_path)

    # Select run 1 — test DB has both enabled
    gui._on_run_item_selected(1)

    val = gui._analysis_wdg.value()
    assert val.enable_calcium is True
    assert val.enable_spikes is True

    gui.close()
