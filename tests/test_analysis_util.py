from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from cali.analysis._fov_metrics import (
    create_stimulation_mask,
    get_overlap_roi_with_stimulated_area,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_create_stimulation_mask_binary(tmp_path: Path) -> None:
    """Test create_stimulation_mask with an already binary image."""
    # Create a binary image (0 and 1)
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:40, 20:40] = 1

    file_path = tmp_path / "binary_mask.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))
    np.testing.assert_array_equal(mask, img)


def test_create_stimulation_mask_full_fov(tmp_path: Path) -> None:
    """Test create_stimulation_mask with a full FOV illumination (all 1s)."""
    img = np.ones((100, 100), dtype=np.uint8)

    file_path = tmp_path / "full_fov.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))
    np.testing.assert_array_equal(mask, img)


def test_create_stimulation_mask_grayscale(tmp_path: Path) -> None:
    """Test create_stimulation_mask with a grayscale image that needs thresholding."""
    # Create a grayscale image with noise
    img = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    # Add a bright spot
    img[30:50, 30:50] = 200

    file_path = tmp_path / "grayscale.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))

    # Check that the mask is binary
    assert np.array_equal(np.unique(mask), [0, 1])
    # Check that the bright spot is detected (roughly)
    assert mask[40, 40] == 1
    assert mask[10, 10] == 0


def test_get_overlap_roi_with_stimulated_area() -> None:
    """Test get_overlap_roi_with_stimulated_area."""
    stim_mask = np.zeros((100, 100), dtype=np.uint8)
    stim_mask[20:60, 20:60] = 1

    # Case 1: ROI completely inside stimulation area
    roi_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask[30:40, 30:40] = 1
    overlap = get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)
    assert overlap == 1.0

    # Case 2: ROI completely outside
    roi_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask[80:90, 80:90] = 1
    overlap = get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)
    assert overlap == 0.0

    # Case 3: ROI partially overlapping (50%)
    roi_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask[50:70, 50:60] = 1  # 20 rows, 10 cols = 200 pixels
    # Stim mask is 20:60, 20:60.
    # Overlap is 50:60 (10 rows), 50:60 (10 cols) = 100 pixels.
    # Fraction should be 100 / 200 = 0.5

    overlap = get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)
    assert overlap == 0.5


def test_get_overlap_roi_with_stimulated_area_empty_roi() -> None:
    """Test with empty ROI mask."""
    stim_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask = np.zeros((100, 100), dtype=np.uint8)

    overlap = get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)
    assert overlap == 0.0


def test_get_overlap_roi_with_stimulated_area_mismatch_shape() -> None:
    """Test with mismatched shapes."""
    stim_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="must have the same dimensions"):
        get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)


# ==================== Additional edge case tests ====================


@pytest.mark.parametrize(
    "roi_coords,stim_coords,expected_overlap",
    [
        # ROI and stimulation are identical
        (
            (slice(20, 40), slice(20, 40)),
            (slice(20, 40), slice(20, 40)),
            1.0,
        ),
        # ROI is subset of stimulation
        (
            (slice(25, 35), slice(25, 35)),
            (slice(20, 40), slice(20, 40)),
            1.0,
        ),
        # Stimulation is subset of ROI (partial overlap)
        (
            (slice(20, 50), slice(20, 50)),
            (slice(30, 40), slice(30, 40)),
            100.0 / 900.0,  # 10*10 / 30*30
        ),
        # No overlap at all
        (
            (slice(10, 20), slice(10, 20)),
            (slice(50, 60), slice(50, 60)),
            0.0,
        ),
        # Edge touching (no actual overlap)
        (
            (slice(10, 20), slice(10, 20)),
            (slice(20, 30), slice(20, 30)),
            0.0,
        ),
    ],
)
def test_get_overlap_parametrized(
    roi_coords: tuple, stim_coords: tuple, expected_overlap: float
) -> None:
    """Parametrized test for various overlap scenarios."""
    stim_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask = np.zeros((100, 100), dtype=np.uint8)

    stim_mask[stim_coords] = 1
    roi_mask[roi_coords] = 1

    overlap = get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)
    assert abs(overlap - expected_overlap) < 1e-6


def test_create_stimulation_mask_various_noise_levels(tmp_path: Path) -> None:
    """Test create_stimulation_mask with different noise levels."""
    # Low noise - should detect bright region clearly
    img = np.random.randint(0, 30, (100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 250
    file_path = tmp_path / "low_noise.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))
    assert np.array_equal(np.unique(mask), [0, 1])
    # Center should be detected
    assert mask[50, 50] == 1


def test_create_stimulation_mask_multiple_regions(tmp_path: Path) -> None:
    """Test create_stimulation_mask with multiple bright regions."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Two separate bright regions
    img[20:30, 20:30] = 200
    img[70:80, 70:80] = 200

    file_path = tmp_path / "multi_region.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))
    assert mask.dtype == np.uint8
    # Both regions should be detected
    assert mask[25, 25] == 1
    assert mask[75, 75] == 1


def test_create_stimulation_mask_edge_region(tmp_path: Path) -> None:
    """Test create_stimulation_mask with bright region at edge."""
    img = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    # Bright region at corner
    img[0:20, 0:20] = 250

    file_path = tmp_path / "edge_region.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))
    assert mask.dtype == np.uint8
    # Should handle edge regions
    assert mask[10, 10] == 1


def test_create_stimulation_mask_uniform_dark(tmp_path: Path) -> None:
    """Test create_stimulation_mask with uniformly dark image."""
    img = np.ones((100, 100), dtype=np.uint8) * 10

    file_path = tmp_path / "uniform_dark.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))
    # Result should be all zeros or all ones depending on threshold
    # At minimum, should not crash and return valid mask
    assert mask.dtype == np.uint8
    assert mask.shape == (100, 100)


def test_get_overlap_complex_shapes() -> None:
    """Test get_overlap with complex non-rectangular shapes."""
    stim_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask = np.zeros((100, 100), dtype=np.uint8)

    # Create circular stimulation area
    y, x = np.ogrid[:100, :100]
    stim_circle = (x - 50) ** 2 + (y - 50) ** 2 <= 20**2
    stim_mask[stim_circle] = 1

    # Create circular ROI completely inside
    roi_circle = (x - 50) ** 2 + (y - 50) ** 2 <= 10**2
    roi_mask[roi_circle] = 1

    overlap = get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)
    # ROI should be completely inside stimulation
    assert overlap == 1.0


def test_get_overlap_partial_circular() -> None:
    """Test get_overlap with partially overlapping circles."""
    stim_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask = np.zeros((100, 100), dtype=np.uint8)

    y, x = np.ogrid[:100, :100]

    # Stimulation circle centered at (40, 50)
    stim_circle = (x - 40) ** 2 + (y - 50) ** 2 <= 15**2
    stim_mask[stim_circle] = 1

    # ROI circle centered at (60, 50) - partially overlapping
    roi_circle = (x - 60) ** 2 + (y - 50) ** 2 <= 15**2
    roi_mask[roi_circle] = 1

    overlap = get_overlap_roi_with_stimulated_area(stim_mask, roi_mask)
    # Should be partial overlap
    assert 0.0 < overlap < 1.0


def test_create_stimulation_mask_already_binary_zeros(tmp_path: Path) -> None:
    """Test create_stimulation_mask with binary mask that's all zeros."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:40, 20:40] = 1

    file_path = tmp_path / "binary_zeros.tif"
    tifffile.imwrite(file_path, img)

    mask = create_stimulation_mask(str(file_path))
    # Should return the same binary mask
    np.testing.assert_array_equal(mask, img)


def test_compute_fov_analysis_with_active_rois() -> None:
    """Test compute_fov_analysis with active ROIs that have traces and peaks."""
    from cali.analysis._fov_analysis import compute_fov_analysis
    from cali.sqlmodel import (
        FOV,
        ROI,
        AnalysisSettings,
        DataAnalysis,
        Traces,
    )

    # Create analysis settings
    analysis_settings = AnalysisSettings(
        peaks_prominence_multiplier=3.0,
    )

    # Create a FOV with active ROIs
    fov = FOV(position_index=0, name="test_fov")

    # Create synthetic traces with peaks at known locations
    num_timepoints = 100
    for i in range(1, 4):  # 3 ROIs
        # Create den_dff with peaks at specific locations for this ROI
        den_dff = np.zeros(num_timepoints)
        # Add peaks at different times for each ROI (shifted slightly)
        peak_times = [10 + i * 2, 30 + i * 2, 50 + i * 2, 70 + i * 2]
        for pt in peak_times:
            if pt < num_timepoints:
                den_dff[pt] = 1.0

        # Create spikes at the same locations
        spikes = np.zeros(num_timepoints)
        for pt in peak_times:
            if pt < num_timepoints:
                spikes[pt] = 1.0

        traces = Traces(
            raw=np.random.randn(num_timepoints).tolist(),
            dff=np.random.randn(num_timepoints).tolist(),
            den_dff=den_dff.tolist(),
            inferred_spikes=spikes.tolist(),
        )

        data_analysis = DataAnalysis(
            total_recording_time_sec=10.0,
            den_dff_frequency=0.4,
            peaks_den_dff=peak_times,
            peaks_amplitudes_den_dff=[1.0] * len(peak_times),
            iei=[0.25] * (len(peak_times) - 1),
            inferred_spikes_threshold=0.5,  # Required for spike analysis
        )

        roi = ROI(label_value=i, active=True)
        roi._new_traces = [traces]
        roi._new_data_analysis = [data_analysis]
        fov.rois.append(roi)

    # Compute FOV analysis
    result = compute_fov_analysis(fov, analysis_settings)

    assert result is not None
    assert result.fov_id is None  # Not yet persisted
    assert result.active_roi_labels == [1, 2, 3]

    # Check calcium correlation matrices are computed
    assert result.calcium_dff_correlation_matrix is not None
    assert len(result.calcium_dff_correlation_matrix) == 3
    assert len(result.calcium_dff_correlation_matrix[0]) == 3

    # Check spike synchrony matrices are computed
    assert result.spike_jitter_synchrony_matrix is not None
    assert len(result.spike_jitter_synchrony_matrix) == 3

    # Check spike matrices
    assert result.spike_max_lag_correlation_matrix is not None
    assert result.spike_jitter_synchrony_matrix is not None

    # Check global synchrony values are reasonable
    assert result.global_spike_jitter_synchrony is not None
    assert 0 <= result.global_spike_jitter_synchrony <= 1


def test_compute_fov_analysis_insufficient_rois() -> None:
    """Test compute_fov_analysis returns None with fewer than 2 active ROIs."""
    from cali.analysis._fov_analysis import compute_fov_analysis
    from cali.sqlmodel import FOV, ROI, AnalysisSettings, Traces

    analysis_settings = AnalysisSettings(peaks_prominence_multiplier=3.0)

    # Create FOV with only 1 active ROI
    fov = FOV(position_index=0, name="test_fov")
    roi = ROI(label_value=1, active=True)
    traces = Traces(
        raw=np.random.randn(100).tolist(),
        dff=np.random.randn(100).tolist(),
        den_dff=np.random.randn(100).tolist(),
    )
    roi._new_traces = [traces]
    fov.rois.append(roi)

    result = compute_fov_analysis(fov, analysis_settings)
    assert result is None


def test_compute_fov_analysis_no_active_rois() -> None:
    """Test compute_fov_analysis returns None when no ROIs are active."""
    from cali.analysis._fov_analysis import compute_fov_analysis
    from cali.sqlmodel import FOV, ROI, AnalysisSettings, Traces

    analysis_settings = AnalysisSettings(peaks_prominence_multiplier=3.0)

    # Create FOV with inactive ROIs
    fov = FOV(position_index=0, name="test_fov")
    for i in range(3):
        roi = ROI(label_value=i + 1, active=False)
        traces = Traces(
            raw=np.random.randn(100).tolist(),
            dff=np.random.randn(100).tolist(),
            den_dff=np.random.randn(100).tolist(),
        )
        roi._new_traces = [traces]
        fov.rois.append(roi)

    result = compute_fov_analysis(fov, analysis_settings)
    assert result is None


# ==================== Population Burst Detection Tests ====================


def test_detect_population_bursts_basic() -> None:
    """Test basic population burst detection with spike trains."""
    from cali.analysis._fov_metrics import _detect_spikes_population_bursts

    # Create spike trains with clear bursts
    # 3 ROIs, 200 frames, frame_rate=10 Hz
    frame_rate = 10.0
    n_frames = 200

    # Create synchronized bursts at frames 20-40, 80-100, 140-160
    spike_trains = []
    for _ in range(3):
        train = np.zeros(n_frames)
        train[20:40] = 1.0  # First burst
        train[80:100] = 1.0  # Second burst
        train[140:160] = 1.0  # Third burst
        spike_trains.append(train)

    (
        burst_count,
        avg_duration,
        avg_interval,
        burst_starts,
        burst_ends,
        pop_activity,
        raw_activity,
    ) = _detect_spikes_population_bursts(
        spike_trains=spike_trains,
        frame_rate=frame_rate,
        burst_threshold_percent=50.0,  # 50% threshold
        min_duration_ms=100.0,  # 0.1 sec = 1 frame at 10 Hz
        gaussian_sigma_sec=0.1,  # Minimal smoothing
    )

    assert burst_count == 3
    assert avg_duration is not None
    assert avg_duration > 0
    assert avg_interval is not None
    assert avg_interval > 0
    # Check that burst timings and traces are returned
    assert len(burst_starts) == 3
    assert len(burst_ends) == 3
    assert pop_activity is not None
    assert raw_activity is not None


def test_detect_population_bursts_no_bursts() -> None:
    """Test population burst detection with no bursts."""
    from cali.analysis._fov_metrics import _detect_spikes_population_bursts

    # Create sparse spike trains that don't form population bursts
    spike_trains = []
    for i in range(3):
        train = np.zeros(200)
        train[10 + i * 50] = 1.0  # Individual spikes, not synchronized
        spike_trains.append(train)

    (
        burst_count,
        avg_duration,
        avg_interval,
        burst_starts,
        burst_ends,
        pop_activity,
        raw_activity,
    ) = _detect_spikes_population_bursts(
        spike_trains=spike_trains,
        frame_rate=10.0,
        burst_threshold_percent=80.0,  # High threshold
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.1,
    )

    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None
    assert burst_starts == []
    assert burst_ends == []
    assert pop_activity is not None
    assert raw_activity is not None


def test_detect_population_bursts_insufficient_rois() -> None:
    """Test population burst detection with < 2 ROIs."""
    from cali.analysis._fov_metrics import _detect_spikes_population_bursts

    spike_trains = [np.ones(100)]  # Only 1 ROI

    (
        burst_count,
        avg_duration,
        avg_interval,
        burst_starts,
        burst_ends,
        pop_activity,
        smoothed,
    ) = _detect_spikes_population_bursts(
        spike_trains=spike_trains,
        frame_rate=10.0,
        burst_threshold_percent=50.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.1,
    )

    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None
    assert burst_starts == []
    assert burst_ends == []
    assert pop_activity is None
    assert smoothed is None


def test_detect_population_bursts_min_duration_filter() -> None:
    """Test that short bursts are filtered by minimum duration."""
    from cali.analysis._fov_metrics import _detect_spikes_population_bursts

    frame_rate = 10.0
    n_frames = 200

    # Create short bursts (5 frames = 0.5 sec) and long bursts (20 frames = 2 sec)
    spike_trains = []
    for _ in range(3):
        train = np.zeros(n_frames)
        train[20:25] = 1.0  # Short burst: 5 frames = 500ms
        train[80:100] = 1.0  # Long burst: 20 frames = 2000ms
        spike_trains.append(train)

    # Set min_duration to 1000ms - should only detect the long burst
    (
        burst_count,
        avg_duration,
        avg_interval,
        burst_starts,
        burst_ends,
        _,
        _,
    ) = _detect_spikes_population_bursts(
        spike_trains=spike_trains,
        frame_rate=frame_rate,
        burst_threshold_percent=50.0,
        min_duration_ms=1000.0,
        gaussian_sigma_sec=0.1,
    )

    assert burst_count == 1
    assert avg_duration is not None
    assert avg_duration >= 2.0  # Long burst is 2 seconds
    assert avg_interval is None  # Only 1 burst, no intervals
    assert len(burst_starts) == 1
    assert len(burst_ends) == 1


def test_detect_calcium_population_bursts_basic() -> None:
    """Test basic population burst detection with denoised df/f traces."""
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    # Create denoised df/f traces with clear bursts
    # 3 ROIs, 200 frames, frame_rate=10 Hz
    frame_rate = 10.0
    n_frames = 200

    # Create synchronized activity bursts at frames 20-40, 80-100, 140-160
    peak_events = []
    for _ in range(3):
        trace = np.zeros(n_frames)
        trace[20:40] = 1.0  # First burst - high activity
        trace[80:100] = 1.0  # Second burst
        trace[140:160] = 1.0  # Third burst
        peak_events.append(trace)

    (
        burst_count,
        avg_duration,
        avg_interval,
        _,  # burst_starts
        _,  # burst_ends
        _,  # pop_activity
        _,  # smoothed
    ) = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=frame_rate,
        burst_threshold_percent=60.0,  # 60% of normalized max
        min_duration_ms=100.0,  # 0.1 sec = 1 frame at 10 Hz
        gaussian_sigma_sec=0.1,  # Minimal smoothing
    )

    assert burst_count == 3
    assert avg_duration is not None
    assert avg_duration > 0
    assert avg_interval is not None
    assert avg_interval > 0


def test_detect_calcium_population_bursts_no_bursts() -> None:
    """Test calcium burst detection with no bursts."""
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    # Create traces with very brief activity that's filtered out by min_duration
    # The key is to have activity that's above threshold but too short to be a burst
    np.random.seed(42)
    peak_events = []
    for _i in range(3):
        # Create baseline with brief spikes that are too short (< min_duration)
        trace = np.zeros(200)
        # Add very brief  spikes (2-3 frames each, but min_duration requires 5 frames)
        trace[50:52] = 1.0  # 2 frames - too short
        trace[100:102] = 1.0  # 2 frames - too short
        trace[150:152] = 1.0  # 2 frames - too short
        peak_events.append(trace)

    (
        burst_count,
        avg_duration,
        avg_interval,
        _,  # burst_starts
        _,  # burst_ends
        _,  # pop_activity
        _,  # smoothed
    ) = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=10.0,
        burst_threshold_percent=50.0,  # Moderate threshold
        min_duration_ms=500.0,  # 5 frames minimum (500ms / 100ms per frame)
        gaussian_sigma_sec=0.1,
    )

    # All bursts should be filtered out due to insufficient duration
    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None


def test_detect_calcium_population_bursts_insufficient_rois() -> None:
    """Test calcium burst detection with < 2 ROIs."""
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    peak_events = [np.ones(100)]  # Only 1 ROI

    (
        burst_count,
        avg_duration,
        avg_interval,
        _,  # burst_starts
        _,  # burst_ends
        _,  # pop_activity
        _,  # smoothed
    ) = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=10.0,
        burst_threshold_percent=50.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.1,
    )

    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None


def test_detect_calcium_population_bursts_constant_activity() -> None:
    """Test calcium burst detection with brief spikes too short to be bursts."""
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    # Create traces with brief spikes that are too short after smoothing
    # At 10fps, 1 second = 10 frames, so min_duration=1000ms requires 10 frames
    peak_events = []
    for _ in range(3):
        trace = np.zeros(200)
        # Add 3-frame spike - still too short for min_duration of 1000ms (10 frames)
        trace[100:103] = 1.0  # 3 frames @ 10fps = 300ms, < 1000ms required
        peak_events.append(trace)

    (
        burst_count,
        avg_duration,
        avg_interval,
        _,  # burst_starts
        _,  # burst_ends
        _,  # pop_activity
        _,  # smoothed
    ) = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=10.0,
        burst_threshold_percent=50.0,
        min_duration_ms=1000.0,  # 10 frames at 10fps = 1000ms required
        gaussian_sigma_sec=0.05,  # Small sigma to minimize spreading
    )

    # Burst too short, should be filtered out
    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None


def test_detect_calcium_population_bursts_normalization() -> None:
    """Test that burst detection properly normalizes traces before thresholding."""
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    frame_rate = 10.0
    n_frames = 200

    # Create traces with same burst pattern
    peak_events = []
    for _ in range(3):
        trace = np.zeros(n_frames)
        trace[40:60] = 1.0  # Burst for each ROI
        peak_events.append(trace)

    # Should still detect the synchronized burst
    (
        burst_count,
        avg_duration,
        _avg_interval,
        _,  # burst_starts
        _,  # burst_ends
        _,  # pop_activity
        _,  # smoothed
    ) = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=frame_rate,
        burst_threshold_percent=50.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.5,
    )

    assert burst_count >= 1
    assert avg_duration is not None


def test_detect_calcium_population_bursts_min_duration_filter() -> None:
    """Test that short calcium bursts are filtered by minimum duration."""
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    frame_rate = 10.0
    n_frames = 200

    # Create short bursts (5 frames = 0.5 sec) and long bursts (20 frames = 2 sec)
    peak_events = []
    for _ in range(3):
        trace = np.zeros(n_frames)
        trace[20:25] = 1.0  # Short burst: 5 frames = 500ms
        trace[80:100] = 1.0  # Long burst: 20 frames = 2000ms
        peak_events.append(trace)

    # Set min_duration to 1000ms - should only detect the long burst
    (
        burst_count,
        avg_duration,
        avg_interval,
        _,  # burst_starts
        _,  # burst_ends
        _,  # pop_activity
        _,  # smoothed
    ) = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=frame_rate,
        burst_threshold_percent=50.0,
        min_duration_ms=1000.0,
        gaussian_sigma_sec=0.1,
    )

    assert burst_count == 1
    assert avg_duration is not None
    assert avg_duration >= 1.5  # Long burst is 2 seconds
    assert avg_interval is None  # Only 1 burst, no intervals


def test_detect_population_bursts_edge_cases() -> None:
    """Test population burst detection edge cases."""
    from cali.analysis._fov_metrics import _detect_spikes_population_bursts

    frame_rate = 10.0

    # Burst starting at frame 0
    spike_trains = []
    for _ in range(3):
        train = np.zeros(100)
        train[0:20] = 1.0
        spike_trains.append(train)

    burst_count, _, _, _, _, _, _ = _detect_spikes_population_bursts(
        spike_trains=spike_trains,
        frame_rate=frame_rate,
        burst_threshold_percent=50.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.1,
    )
    assert burst_count >= 1

    # Burst ending at last frame
    spike_trains = []
    for _ in range(3):
        train = np.zeros(100)
        train[80:] = 1.0
        spike_trains.append(train)

    burst_count, _, _, _, _, _, _ = _detect_spikes_population_bursts(
        spike_trains=spike_trains,
        frame_rate=frame_rate,
        burst_threshold_percent=50.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.1,
    )
    assert burst_count >= 1


def test_detect_calcium_population_bursts_absolute_threshold() -> None:
    """Test that calcium burst detection uses an absolute threshold.

    The threshold is burst_threshold_percent / 100, i.e. a fraction of ROIs
    that must have simultaneous peaks, NOT scaled by the max population activity.
    """
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    n_frames = 200
    frame_rate = 10.0

    # 4 ROIs: only 1 out of 4 is active at frames 50-70 → fraction = 0.25
    peak_events = [np.zeros(n_frames) for _ in range(4)]
    peak_events[0][50:70] = 1.0  # Only ROI 0 is active

    # Threshold at 20% (0.20) — should detect the burst (0.25 > 0.20)
    burst_count_low, _, _, _, _, raw, _smoothed = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=frame_rate,
        burst_threshold_percent=20.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.0,
    )
    assert burst_count_low >= 1, "25% activity should exceed 20% threshold"
    # Raw population activity should be 0.25 during the active window
    assert raw is not None
    assert abs(float(np.max(raw)) - 0.25) < 1e-6

    # Threshold at 30% (0.30) — should NOT detect (0.25 < 0.30)
    burst_count_high, _, _, _, _, _, _ = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=frame_rate,
        burst_threshold_percent=30.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.0,
    )
    assert burst_count_high == 0, "25% activity should not exceed 30% threshold"


def test_detect_calcium_population_bursts_returns_raw_before_smoothed() -> None:
    """Test that return order is (raw, smoothed) not (smoothed, raw)."""
    from cali.analysis._fov_metrics import _detect_calcium_population_bursts

    n_frames = 100
    peak_events = []
    for _ in range(3):
        trace = np.zeros(n_frames)
        trace[30:50] = 1.0
        peak_events.append(trace)

    _, _, _, _, _, raw, smoothed = _detect_calcium_population_bursts(
        peak_events=peak_events,
        frame_rate=10.0,
        burst_threshold_percent=50.0,
        min_duration_ms=100.0,
        gaussian_sigma_sec=0.5,
    )
    assert raw is not None and smoothed is not None
    # Raw should have sharp edges; smoothed should be wider
    raw_nonzero = np.count_nonzero(raw > 0.01)
    smoothed_nonzero = np.count_nonzero(smoothed > 0.01)
    assert smoothed_nonzero >= raw_nonzero, (
        "Smoothed trace should be at least as wide as raw"
    )


# ---------------------------------------------------------------------------
# _get_fraction_significant_pairs
# ---------------------------------------------------------------------------


def test_get_fraction_significant_pairs_none() -> None:
    from cali.analysis._fov_metrics import _get_fraction_significant_pairs

    assert _get_fraction_significant_pairs(None) is None


def test_get_fraction_significant_pairs_empty() -> None:
    from cali.analysis._fov_metrics import _get_fraction_significant_pairs

    assert _get_fraction_significant_pairs(np.array([])) is None


def test_get_fraction_significant_pairs_1x1() -> None:
    from cali.analysis._fov_metrics import _get_fraction_significant_pairs

    assert _get_fraction_significant_pairs(np.array([[3.0]])) is None


def test_get_fraction_significant_pairs_nonsquare() -> None:
    from cali.analysis._fov_metrics import _get_fraction_significant_pairs

    assert _get_fraction_significant_pairs(np.ones((2, 3))) is None


def test_get_fraction_significant_pairs_valid() -> None:
    from cali.analysis._fov_metrics import _get_fraction_significant_pairs

    mat = np.array([[0.0, 3.0], [3.0, 0.0]])
    result = _get_fraction_significant_pairs(mat, threshold=2.0)
    assert result == 1.0

    mat2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    result2 = _get_fraction_significant_pairs(mat2, threshold=2.0)
    assert result2 == 0.0
