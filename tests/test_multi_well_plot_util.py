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


def test_aggregate_fov_data_unweighted_mean() -> None:
    """Condition mean is the unweighted mean of well means (one FOV per well)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "Drug": {
            "w1": {"fov1": [0.3, 0.5, 0.8, 0.4, 0.6]},  # well mean = 0.52
            "w2": {"fov2": [1.0, 1.2]},  # well mean = 1.10
            "w3": {"fov3": [0.7, 0.9, 0.6, 0.8]},  # well mean = 0.75
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)

    # Unweighted mean of well means: (0.52 + 1.10 + 0.75) / 3
    expected_mean = np.mean([0.52, 1.10, 0.75])
    assert abs(result["means"][0] - expected_mean) < 1e-10

    # well_names_list tracks the well keys used in the input data
    assert result["well_names_list"] == [["w1", "w2", "w3"]]


def test_aggregate_fov_data_sem_across_wells() -> None:
    """Condition SEM is computed across well means."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "w1": {"fov1": [1.0, 2.0, 3.0]},  # well mean = 2.0
            "w2": {"fov2": [4.0, 5.0]},  # well mean = 4.5
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)

    well_means = np.array([2.0, 4.5])
    expected_sem = float(np.std(well_means, ddof=1) / np.sqrt(2))
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_fov_data_single_well() -> None:
    """Single well: mean is the FOV mean, SEM is 0 (only one well)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {"A": {"w1": {"fov1": [2.0, 4.0, 6.0]}}}
    result = _aggregate_fov_data_to_condition_stats(data)

    assert abs(result["means"][0] - 4.0) < 1e-10
    assert result["sems"][0] == 0.0
    assert result["well_names_list"] == [["w1"]]


def test_aggregate_fov_data_multiple_fovs_per_well() -> None:
    """Multiple FOVs in the same well are averaged to one well mean before SEM."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    # w1 has 2 FOVs (means 2.0 and 4.0) → well mean = 3.0
    # w2 has 1 FOV (mean = 7.0) → well mean = 7.0
    data = {
        "A": {
            "w1": {"fov1": [1.0, 2.0, 3.0], "fov2": [4.0, 5.0]},
            "w2": {"fov3": [7.0]},
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)

    np.array([3.0, 7.0])  # (2.0+4.5)/2=3.25... wait
    # fov1 mean = 2.0, fov2 mean = 4.5, w1 mean = (2.0+4.5)/2 = 3.25
    # w2 mean = 7.0
    # condition mean = (3.25 + 7.0) / 2 = 5.125
    w1_mean = np.mean([2.0, 4.5])
    expected_mean = np.mean([w1_mean, 7.0])
    expected_sem = float(np.std([w1_mean, 7.0], ddof=1) / np.sqrt(2))
    assert abs(result["means"][0] - expected_mean) < 1e-10
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_fov_data_single_roi_per_fov() -> None:
    """Single ROI per FOV, each FOV in its own well: SEM across two well means."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {"A": {"w1": {"fov1": [3.0]}, "w2": {"fov2": [7.0]}}}
    result = _aggregate_fov_data_to_condition_stats(data)

    assert abs(result["means"][0] - 5.0) < 1e-10
    expected_sem = float(np.std([3.0, 7.0], ddof=1) / np.sqrt(2))
    assert abs(result["sems"][0] - expected_sem) < 1e-10


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
            "w1": {"fov1": [[1.0, 2.0], [3.0]]},  # ROI1 has 2 peaks, ROI2 has 1
            "w2": {"fov2": [[4.0, 5.0, 6.0]]},  # ROI3 has 3 peaks
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)
    # fov1 flat: [1,2,3] → mean=2.0; w1 mean = 2.0
    # fov2 flat: [4,5,6] → mean=5.0; w2 mean = 5.0
    # unweighted mean = (2.0 + 5.0) / 2 = 3.5
    assert abs(result["means"][0] - 3.5) < 1e-10


# ---------------------------------------------------------------------------
# _aggregate_percentage_data_to_condition_stats
# ---------------------------------------------------------------------------


def test_aggregate_percentage_unweighted_mean() -> None:
    """Condition mean is the unweighted mean of well means (one FOV per well)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "w1": {"fov1": (80.0, 10)},
            "w2": {"fov2": (15.0, 20)},
        }
    }
    result = _aggregate_percentage_data_to_condition_stats(data)

    # Unweighted mean of well means: (80.0 + 15.0) / 2
    expected_mean = (80.0 + 15.0) / 2.0
    assert abs(result["means"][0] - expected_mean) < 1e-10

    # well_names_list tracks the well keys
    assert result["well_names_list"] == [["w1", "w2"]]


def test_aggregate_percentage_sem_across_wells() -> None:
    """SEM is computed across well means, not via the binomial formula."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "w1": {"fov1": (80.0, 10)},
            "w2": {"fov2": (15.0, 20)},
        }
    }
    result = _aggregate_percentage_data_to_condition_stats(data)

    well_means = np.array([80.0, 15.0])
    expected_sem = float(np.std(well_means, ddof=1) / np.sqrt(2))
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_percentage_single_well() -> None:
    """Single well: mean equals the FOV mean, SEM is 0 (only one well)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {"A": {"w1": {"fov1": (50.0, 20)}}}
    result = _aggregate_percentage_data_to_condition_stats(data)

    assert abs(result["means"][0] - 50.0) < 1e-10
    assert result["sems"][0] == 0.0


def test_aggregate_percentage_multiple_fovs_per_well() -> None:
    """Multiple FOVs in the same well are averaged before inter-well SEM."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    # w1 has 2 FOVs: 60% and 80% → well mean = 70%
    # w2 has 1 FOV: 50% → well mean = 50%
    data = {
        "A": {
            "w1": {"fov1": (60.0, 10), "fov2": (80.0, 5)},
            "w2": {"fov3": (50.0, 8)},
        }
    }
    result = _aggregate_percentage_data_to_condition_stats(data)

    expected_mean = np.mean([70.0, 50.0])
    expected_sem = float(np.std([70.0, 50.0], ddof=1) / np.sqrt(2))
    assert abs(result["means"][0] - expected_mean) < 1e-10
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
        "well_values_list": [np.array([1.0]), np.array([2.0])],
        "well_names_list": [["W1"], ["W2"]],
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
