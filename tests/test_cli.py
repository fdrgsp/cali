"""Tests for the cali CLI (__main__.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cali.__main__ import _str_to_bool, main

TEST_DB = Path(__file__).parent / "test_data" / "data_and_db_for_tests" / "test_db.cali"


# ── _str_to_bool helper ──────────────────────────────────────────────────────


def test_str_to_bool_true_values() -> None:
    for v in ("True", "true", "TRUE", "1", "yes", "YES"):
        assert _str_to_bool(v) is True


def test_str_to_bool_false_values() -> None:
    for v in ("False", "false", "FALSE", "0", "no", "NO"):
        assert _str_to_bool(v) is False


def test_str_to_bool_invalid_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _str_to_bool("maybe")


# ── tree sub-command: argument parsing ──────────────────────────────────────


def test_tree_default_level_and_settings(capsys: pytest.CaptureFixture) -> None:
    """tree with no optional flags defaults to level=roi, show_settings=True."""
    main(["tree", str(TEST_DB)])
    out = capsys.readouterr().out
    # The rich tree root line shows result count
    assert "Analysis Result" in out
    # Detailed settings are present (show_settings=True by default)
    assert "Cellpose Parameters" in out
    # ROI entries appear in the default (roi) depth
    assert "ROI" in out


def test_tree_level_fov(capsys: pytest.CaptureFixture) -> None:
    """tree -l fov limits depth to FOV level (no ROI rows)."""
    main(["tree", str(TEST_DB), "-l", "fov"])
    out = capsys.readouterr().out
    assert "Analysis Result" in out
    # FOV entries appear
    assert "fov:" in out
    # ROI entries should not appear
    assert "ROI" not in out


def test_tree_show_settings_false(capsys: pytest.CaptureFixture) -> None:
    """tree -s False hides detailed settings parameters."""
    main(["tree", str(TEST_DB), "-s", "False"])
    out = capsys.readouterr().out
    assert "Analysis Result" in out
    # Detailed parameter blocks should be absent
    assert "Cellpose Parameters" not in out
    assert "Neuropil Correction" not in out


def test_tree_long_flags(capsys: pytest.CaptureFixture) -> None:
    """--level and --show-settings long forms produce wells but not FOVs."""
    main(["tree", str(TEST_DB), "--level", "well", "--show-settings", "False"])
    out = capsys.readouterr().out
    assert "Analysis Result" in out
    # Wells appear
    assert "Conditions:" in out
    # FOV and ROI entries should not appear at well depth
    assert "fov:" not in out
    assert "ROI" not in out


def test_tree_missing_db(tmp_path: Path) -> None:
    """tree exits with code 1 when the DB file does not exist."""
    missing = tmp_path / "nonexistent.cali"
    with pytest.raises(SystemExit) as exc_info:
        main(["tree", str(missing)])
    assert exc_info.value.code == 1


def test_tree_invalid_level() -> None:
    """tree rejects an unknown level value."""
    with pytest.raises(SystemExit):
        main(["tree", str(TEST_DB), "-l", "invalid_level"])
