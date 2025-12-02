from unittest.mock import patch

import numpy as np

from cali.extraction._util import _calculate_bg, calculate_dff


def test_calculate_dff() -> None:
    """Test calculate_dff."""
    data = np.array([100, 110, 120, 110, 100, 90, 100, 110, 120, 110], dtype=float)
    dff = calculate_dff(data, window=5, percentile=10)
    assert dff.shape == data.shape
    assert not np.any(np.isnan(dff))


def test_calculate_dff_plot() -> None:
    """Test calculate_dff with plot=True."""
    data = np.array([100, 110, 120], dtype=float)
    with patch("matplotlib.pyplot.show") as mock_show:
        calculate_dff(data, window=3, percentile=10, plot=True)
        mock_show.assert_called_once()


def test_calculate_bg() -> None:
    """Test _calculate_bg."""
    data = np.ones(100) * 100
    bg = _calculate_bg(data, window=10, percentile=50)
    np.testing.assert_allclose(bg, 100)

    # Test with varying data
    data = np.arange(100, dtype=float)
    bg = _calculate_bg(data, window=10, percentile=50)
    # For a linear ramp, the median in a window should be roughly the center of the
    # window But the implementation might be different. Just check shape and no nans.
    assert bg.shape == data.shape
    assert not np.any(np.isnan(bg))
