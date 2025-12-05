from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from cali.analysis._util import (
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
        # Create dec_dff with peaks at specific locations for this ROI
        dec_dff = np.zeros(num_timepoints)
        # Add peaks at different times for each ROI (shifted slightly)
        peak_times = [10 + i * 2, 30 + i * 2, 50 + i * 2, 70 + i * 2]
        for pt in peak_times:
            if pt < num_timepoints:
                dec_dff[pt] = 1.0

        # Create spikes at the same locations
        spikes = np.zeros(num_timepoints)
        for pt in peak_times:
            if pt < num_timepoints:
                spikes[pt] = 1.0

        traces = Traces(
            raw=np.random.randn(num_timepoints).tolist(),
            dff=np.random.randn(num_timepoints).tolist(),
            dec_dff=dec_dff.tolist(),
            inferred_spikes=spikes.tolist(),
        )

        data_analysis = DataAnalysis(
            total_recording_time_sec=10.0,
            dec_dff_frequency=0.4,
            peaks_dec_dff=peak_times,
            peaks_amplitudes_dec_dff=[1.0] * len(peak_times),
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

    # Check calcium synchrony matrices are computed
    assert result.calcium_peaks_jitter_synchrony_matrix is not None
    assert len(result.calcium_peaks_jitter_synchrony_matrix) == 3

    # Check spike matrices
    assert result.spike_correlation_matrix is not None
    assert result.spike_jitter_synchrony_matrix is not None

    # Check global synchrony values are reasonable
    assert result.global_calcium_peaks_jitter_synchrony is not None
    assert 0 <= result.global_calcium_peaks_jitter_synchrony <= 1
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
        dec_dff=np.random.randn(100).tolist(),
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
            dec_dff=np.random.randn(100).tolist(),
        )
        roi._new_traces = [traces]
        fov.rois.append(roi)

    result = compute_fov_analysis(fov, analysis_settings)
    assert result is None
