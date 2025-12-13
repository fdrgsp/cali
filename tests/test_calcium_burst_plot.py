"""Test calcium burst plotting functions."""

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.engine import create_engine

from cali.plot._single_wells_plots.burst._plot_burst_activity import (
    _get_calcium_burst_parameters,
    _get_population_calcium_data,
)


@pytest.fixture
def test_db_path():
    """Path to test database."""
    return Path(__file__).parent / "test_data/data_and_db_for_tests/test_db.cali"


@pytest.fixture
def test_engine(test_db_path):
    """Create test database engine."""
    if not test_db_path.exists():
        pytest.skip(f"Test database not found at {test_db_path}")

    engine = create_engine(
        f"sqlite:///{test_db_path}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )
    return engine


def test_get_calcium_burst_parameters(test_engine) -> None:
    """Test retrieving calcium burst parameters from database."""
    # Test with run_id=1 (should have analysis settings)
    result = _get_calcium_burst_parameters(test_engine, run_id=1)

    assert result is not None, "Should return parameters for valid run_id"
    threshold, min_duration, sigma = result

    # Verify expected values from tests.json
    assert threshold == 50.0
    assert min_duration == 500.0
    assert sigma == 0.5


def test_get_calcium_burst_parameters_no_run_id(test_engine) -> None:
    """Test retrieving calcium burst parameters without run_id."""
    # Test without run_id (should get most recent settings)
    result = _get_calcium_burst_parameters(test_engine, run_id=None)

    assert result is not None, "Should return parameters for most recent settings"
    threshold, min_duration, sigma = result

    # Verify they are valid floats
    assert isinstance(threshold, float)
    assert isinstance(min_duration, float)
    assert isinstance(sigma, float)


def test_get_population_calcium_data(test_engine) -> None:
    """Test extracting population calcium data from database."""
    # Test with first FOV
    fov_name = "B5_0000"

    calcium_traces, roi_names, time_axis = _get_population_calcium_data(
        test_engine, fov_name, rois=None, run_id=1
    )

    # Verify we got data
    assert calcium_traces is not None, "Should return calcium traces"
    assert len(roi_names) > 0, "Should have ROI names"
    assert len(time_axis) > 0, "Should have time axis"

    # Verify shape consistency
    assert calcium_traces.shape[0] == len(roi_names), (
        "Number of traces should match number of ROI names"
    )
    assert calcium_traces.shape[1] == len(time_axis), (
        "Trace length should match time axis length"
    )

    # Verify data types
    assert isinstance(calcium_traces, np.ndarray)
    assert calcium_traces.dtype == np.float64
    assert all(isinstance(name, str) for name in roi_names)
    assert isinstance(time_axis, np.ndarray)


def test_get_population_calcium_data_with_roi_filter(test_engine) -> None:
    """Test extracting population calcium data with ROI filtering."""
    fov_name = "B5_0000"

    # First get all ROIs to know what's available
    all_traces, all_names, _ = _get_population_calcium_data(
        test_engine, fov_name, rois=None, run_id=1
    )

    if all_traces is not None and len(all_names) >= 2:
        # Test with specific ROI subset (use first 2 ROI label values)
        roi_labels = [int(name) for name in all_names[:2]]

        filtered_traces, filtered_names, _filtered_time = _get_population_calcium_data(
            test_engine, fov_name, rois=roi_labels, run_id=1
        )

        assert filtered_traces is not None
        assert len(filtered_names) == 2, "Should have exactly 2 ROIs"
        assert filtered_traces.shape[0] == 2


def test_get_population_calcium_data_invalid_fov(test_engine) -> None:
    """Test with non-existent FOV."""
    calcium_traces, roi_names, time_axis = _get_population_calcium_data(
        test_engine, "INVALID_FOV", rois=None, run_id=1
    )

    assert calcium_traces is None, "Should return None for invalid FOV"
    assert len(roi_names) == 0
    assert len(time_axis) == 0


def test_get_population_calcium_data_too_few_rois(test_engine) -> None:
    """Test with FOV that has less than 2 active ROIs."""
    # This should return None if less than 2 traces are found
    # (burst detection requires at least 2 ROIs for population analysis)
    fov_name = "B5_0000"

    # Try to get data with ROI filter that results in < 2 ROIs
    calcium_traces, _roi_names, _time_axis = _get_population_calcium_data(
        test_engine,
        fov_name,
        rois=[999999],
        run_id=1,  # Non-existent ROI
    )

    # Should handle gracefully
    assert calcium_traces is None or calcium_traces.shape[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
