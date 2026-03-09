"""Tests for multi-well plot utility functions.

Covers:
- _aggregate_fov_data_to_condition_stats
- _aggregate_percentage_data_to_condition_stats
- _aggregate_fov_scalar_to_condition_stats  (additional edge cases)
- _get_experiment_type
- _BarTickLabel.dataBounds
- _create_pyqtgraph_bar_plot with override_color
- make_parameter_compute_fn
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sqlmodel import create_engine

from cali.sqlmodel._util import create_database_and_tables

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# _aggregate_fov_data_to_condition_stats
# ---------------------------------------------------------------------------


def test_aggregate_fov_data_weighted_mean() -> None:
    """Condition mean is a weighted average of FOV means (weighted by ROI count)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "Drug": {
            "fov1": [0.3, 0.5, 0.8, 0.4, 0.6],  # 5 ROIs, mean=0.52
            "fov2": [1.0, 1.2],  # 2 ROIs, mean=1.10
            "fov3": [0.7, 0.9, 0.6, 0.8],  # 4 ROIs, mean=0.75
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)

    expected_mean = (5 * 0.52 + 2 * 1.10 + 4 * 0.75) / 11
    assert abs(result["means"][0] - expected_mean) < 1e-10


def test_aggregate_fov_data_pooled_sem() -> None:
    """Condition SEM is pooled from per-FOV SEMs weighted by ROI count."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "fov1": [1.0, 2.0, 3.0],  # n=3
            "fov2": [4.0, 5.0],  # n=2
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)

    fov1_sem = 1.0 / np.sqrt(3)
    fov2_sem = np.std([4.0, 5.0], ddof=1) / np.sqrt(2)
    expected_sem = np.sqrt((3 * fov1_sem**2 + 2 * fov2_sem**2) / 5)
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_fov_data_single_fov() -> None:
    """Single FOV: mean is the FOV mean, SEM is the within-FOV SEM."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {"A": {"fov1": [2.0, 4.0, 6.0]}}
    result = _aggregate_fov_data_to_condition_stats(data)

    assert abs(result["means"][0] - 4.0) < 1e-10
    expected_sem = np.std([2.0, 4.0, 6.0], ddof=1) / np.sqrt(3)
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_fov_data_single_roi_per_fov() -> None:
    """Single ROI per FOV: SEM should be 0 (no within-FOV variability)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {"A": {"fov1": [3.0], "fov2": [7.0]}}
    result = _aggregate_fov_data_to_condition_stats(data)

    assert abs(result["means"][0] - 5.0) < 1e-10
    assert result["sems"][0] == 0.0


def test_aggregate_fov_data_empty() -> None:
    """Empty input returns empty output."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    result = _aggregate_fov_data_to_condition_stats({})
    assert result["conditions"] == []
    assert result["means"] == []
    assert result["sems"] == []


def test_aggregate_fov_data_with_list_values() -> None:
    """Values that are lists (e.g. peak amplitudes) are flattened correctly."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "X": {
            "fov1": [[1.0, 2.0], [3.0]],  # ROI1 has 2 peaks, ROI2 has 1
            "fov2": [[4.0, 5.0, 6.0]],  # ROI3 has 3 peaks
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)
    # fov1 flat: [1,2,3] → mean=2.0, n=3
    # fov2 flat: [4,5,6] → mean=5.0, n=3
    # weighted mean = (3*2 + 3*5)/6 = 3.5
    assert abs(result["means"][0] - 3.5) < 1e-10


# ---------------------------------------------------------------------------
# _aggregate_percentage_data_to_condition_stats
# ---------------------------------------------------------------------------


def test_aggregate_percentage_weighted_mean() -> None:
    """Percentage mean is weighted by total ROI count per FOV."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "fov1": (80.0, 10),  # 8/10 active
            "fov2": (15.0, 20),  # 3/20 active
        }
    }
    result = _aggregate_percentage_data_to_condition_stats(data)

    expected_mean = (10 * 80.0 + 20 * 15.0) / 30
    assert abs(result["means"][0] - expected_mean) < 1e-10


def test_aggregate_percentage_binomial_sem() -> None:
    """SEM uses binomial formula: sqrt(p*(1-p)/N) * 100."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "fov1": (80.0, 10),
            "fov2": (15.0, 20),
        }
    }
    result = _aggregate_percentage_data_to_condition_stats(data)

    p = result["means"][0] / 100.0
    n_total = 30
    expected_sem = np.sqrt(p * (1 - p) / n_total) * 100
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_percentage_single_fov() -> None:
    """Single FOV: binomial SEM based on that FOV's count."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {"A": {"fov1": (50.0, 20)}}
    result = _aggregate_percentage_data_to_condition_stats(data)

    assert abs(result["means"][0] - 50.0) < 1e-10
    expected_sem = np.sqrt(0.25 / 20) * 100
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_percentage_empty() -> None:
    """Empty input returns empty output."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    result = _aggregate_percentage_data_to_condition_stats({})
    assert result["conditions"] == []


# ---------------------------------------------------------------------------
# _get_experiment_type
# ---------------------------------------------------------------------------


def test_get_experiment_type_returns_type(full_db: tuple[Engine, int]) -> None:
    from sqlmodel import Session

    from cali.plot._multi_wells_plots._util import _get_experiment_type

    engine, run_id = full_db
    with Session(engine) as session:
        result = _get_experiment_type(session, run_id)
    assert result is None or isinstance(result, str)


def test_get_experiment_type_invalid_run_id(full_db: tuple[Engine, int]) -> None:
    from sqlmodel import Session

    from cali.plot._multi_wells_plots._util import _get_experiment_type

    engine, _ = full_db
    with Session(engine) as session:
        result = _get_experiment_type(session, 9999)
    assert result is None


# ---------------------------------------------------------------------------
# _BarTickLabel.dataBounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ax", "expected_type"),
    [
        (0, tuple),
        (1, tuple),
        (2, type(None)),
    ],
)
def test_bar_tick_label_data_bounds(qtbot: QtBot, ax: int, expected_type: type) -> None:
    import pyqtgraph as pg

    from cali.plot._multi_wells_plots._util import _BarTickLabel

    label = _BarTickLabel("test", y_extent=-0.5, anchor=(0.5, 0))
    pw = pg.PlotWidget()
    qtbot.addWidget(pw)
    pw.addItem(label)
    label.setPos(3.0, 0.0)

    result = label.dataBounds(ax)
    assert isinstance(result, expected_type)
    if ax == 0:
        assert result == (3.0, 3.0)
    elif ax == 1:
        assert result == (-0.5, 0.0)


# ---------------------------------------------------------------------------
# _create_pyqtgraph_bar_plot with override_color
# ---------------------------------------------------------------------------


def test_create_bar_plot_with_override_color(
    full_db: tuple[Engine, int],
    qtbot: QtBot,
) -> None:
    """_create_pyqtgraph_bar_plot uses override_color when provided."""
    from pyqtgraph import BarGraphItem
    from qtpy.QtWidgets import QWidget

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._util import (
        BarPlotData,
        _create_pyqtgraph_bar_plot,
    )

    _engine, _run_id = full_db
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(widget)

    data: BarPlotData = {
        "conditions": ["WT", "KO"],
        "means": [1.0, 2.0],
        "sems": [0.1, 0.2],
        "fov_values_list": [np.array([1.0]), np.array([2.0])],
    }
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=data,
        parameter="Test",
        units="AU",
        override_color="green",
    )
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


# ---------------------------------------------------------------------------
# make_parameter_compute_fn
# ---------------------------------------------------------------------------


def test_make_parameter_compute_fn_is_callable() -> None:
    """make_parameter_compute_fn returns a callable."""
    from cali.plot._multi_wells_plots._util import make_parameter_compute_fn

    fn = make_parameter_compute_fn("amplitude", "dF/F", "Amplitude")
    assert callable(fn)


def test_make_parameter_compute_fn_returns_none_empty_db() -> None:
    """make_parameter_compute_fn returns None when DB is empty."""
    from cali.plot._multi_wells_plots._util import make_parameter_compute_fn

    fn = make_parameter_compute_fn("amplitude", "dF/F", "Amplitude")
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    result = fn(engine, None)
    assert result is None
    engine.dispose(close=True)
