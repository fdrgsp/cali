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
