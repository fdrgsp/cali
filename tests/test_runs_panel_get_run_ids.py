"""Tests for RunsPanel.get_run_ids() method."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cali.gui._runs_panel import _RunsPanel

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    pass


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
