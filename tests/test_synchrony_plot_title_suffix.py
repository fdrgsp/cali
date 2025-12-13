"""Tests for synchrony plot title_suffix parameter.

This module tests that the title_suffix parameter correctly appends
to plot titles in synchrony plotting functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import Session, create_engine, select

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony import (
    _plot_spike_synchrony_data,
)
from cali.sqlmodel import FOV

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot


@pytest.fixture
def widget(qtbot: QtBot) -> _SingleWellGraphWidget:
    """Create a _SingleWellGraphWidget for testing."""
    widget = _SingleWellGraphWidget(None)  # type: ignore[arg-type]
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def test_db_path() -> Path:
    """Return path to test database."""
    from pathlib import Path

    return (
        Path(__file__).parent / "test_data" / "data_and_db_for_tests" / "test_db.cali"
    )


def test_spike_synchrony_title_suffix_with_data(
    widget: _SingleWellGraphWidget,
    test_db_path: Path,
    qtbot: QtBot,
) -> None:
    """Test that title_suffix is appended to spike synchrony plot titles."""
    engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        with Session(engine) as session:
            fov = session.exec(select(FOV).limit(1)).first()
            assert fov is not None
            fov_name = fov.name

        # Plot with title_suffix
        _plot_spike_synchrony_data(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=None,
            run_id=1,
            title_suffix=" - Non-Stimulated",
        )

        # Verify suffix appears in title
        plot_title = widget.plot_item.titleLabel.text  # type: ignore[union-attr]
        assert " - Non-Stimulated" in plot_title, (
            f"Expected suffix in title, got: {plot_title}"
        )
    finally:
        engine.dispose(close=True)


def test_spike_synchrony_title_suffix_no_data(
    widget: _SingleWellGraphWidget,
    test_db_path: Path,
    qtbot: QtBot,
) -> None:
    """Test title_suffix appears when there's no spike data."""
    engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        with Session(engine) as session:
            fov = session.exec(select(FOV).limit(1)).first()
            assert fov is not None
            fov_name = fov.name

        # Plot with insufficient ROIs and title_suffix
        _plot_spike_synchrony_data(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=[1],  # Single ROI - insufficient for synchrony
            run_id=1,
            title_suffix=" - Test",
        )

        # Should show "Need ≥2 ROIs" with suffix
        plot_title = widget.plot_item.titleLabel.text  # type: ignore[union-attr]
        assert "Need ≥2 ROIs" in plot_title
        assert " - Test" in plot_title, f"Expected suffix in title, got: {plot_title}"
    finally:
        engine.dispose(close=True)


@pytest.mark.parametrize("suffix", ["", " - NonStim", " (Control)", " [Baseline]"])
def test_spike_synchrony_various_suffixes(
    widget: _SingleWellGraphWidget,
    test_db_path: Path,
    qtbot: QtBot,
    suffix: str,
) -> None:
    """Test spike synchrony with various suffix formats."""
    engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        with Session(engine) as session:
            fov = session.exec(select(FOV).limit(1)).first()
            assert fov is not None
            fov_name = fov.name

        _plot_spike_synchrony_data(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=None,
            run_id=1,
            title_suffix=suffix,
        )

        plot_title = widget.plot_item.titleLabel.text  # type: ignore[union-attr]
        if suffix:
            msg = f"Expected '{suffix}' in title, got: {plot_title}"
            assert suffix in plot_title, msg
        else:
            # Empty suffix should work fine
            assert plot_title, "Title should not be empty"
    finally:
        engine.dispose(close=True)
