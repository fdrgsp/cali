"""Tests for neuropil extraction functions."""

import numpy as np
import pytest

from cali.extraction._neuropil import (
    _create_cell_pix_from_masks,
    _extendROI,
    create_neuropil_from_dilation,
)


def test_extendROI_single_iteration() -> None:
    """Test _extendROI with a single iteration."""
    # Start with a single pixel at (5, 5)
    ypix = np.array([5])
    xpix = np.array([5])
    Ly, Lx = 10, 10

    ypix_ext, xpix_ext = _extendROI(ypix, xpix, Ly, Lx, niter=1)

    # Should expand to include 4 cardinal neighbors: (5,5), (4,5), (6,5), (5,4), (5,6)
    # Total 5 pixels after unique
    assert len(ypix_ext) == 5
    assert len(xpix_ext) == 5

    # Check that all pixels are within bounds
    assert np.all(ypix_ext >= 0)
    assert np.all(ypix_ext < Ly)
    assert np.all(xpix_ext >= 0)
    assert np.all(xpix_ext < Lx)


def test_extendROI_multiple_iterations() -> None:
    """Test _extendROI with multiple iterations."""
    ypix = np.array([5])
    xpix = np.array([5])
    Ly, Lx = 20, 20

    # Two iterations should create a larger region
    ypix_ext, _ = _extendROI(ypix, xpix, Ly, Lx, niter=2)

    # Should have more pixels than single iteration
    assert len(ypix_ext) > 5
    assert np.all(ypix_ext >= 0)
    assert np.all(ypix_ext < Ly)


def test_extendROI_boundary_clipping() -> None:
    """Test _extendROI respects image boundaries."""
    # Start at corner (0, 0)
    ypix = np.array([0])
    xpix = np.array([0])
    Ly, Lx = 10, 10

    ypix_ext, xpix_ext = _extendROI(ypix, xpix, Ly, Lx, niter=1)

    # Should not extend outside bounds
    assert np.all(ypix_ext >= 0)
    assert np.all(ypix_ext < Ly)
    assert np.all(xpix_ext >= 0)
    assert np.all(xpix_ext < Lx)

    # Should have fewer pixels than if started at center
    # (can only expand right and down, not left/up)
    assert len(ypix_ext) == 3  # (0,0), (1,0), (0,1)


@pytest.mark.parametrize(
    "cell_masks,height,width,expected_shape",
    [
        # Test case 1: Single small cell
        (
            [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)],
            3,
            3,
            (3, 3),
        ),
        # Test case 2: Two cells
        (
            [
                np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]], dtype=bool),
            ],
            3,
            3,
            (3, 3),
        ),
        # Test case 3: Empty mask list
        (
            [],
            5,
            5,
            (5, 5),
        ),
    ],
)
def test_create_cell_pix_from_masks(
    cell_masks: list[np.ndarray],
    height: int,
    width: int,
    expected_shape: tuple[int, int],
) -> None:
    """Test _create_cell_pix_from_masks with various inputs."""
    cell_pix = _create_cell_pix_from_masks(cell_masks, height, width)

    assert cell_pix.shape == expected_shape
    assert cell_pix.dtype == np.float32

    # Cell pixels should be in [0, 1] range
    assert np.all(cell_pix >= 0)
    assert np.all(cell_pix <= 1)


def test_create_cell_pix_from_masks_empty_mask() -> None:
    """Test _create_cell_pix_from_masks with a mask containing no pixels."""
    # Create a mask with all zeros
    empty_mask = np.zeros((5, 5), dtype=bool)
    cell_masks = [empty_mask]

    cell_pix = _create_cell_pix_from_masks(cell_masks, 5, 5)

    # Should return all zeros
    assert cell_pix.shape == (5, 5)
    assert np.all(cell_pix == 0)


def test_create_cell_pix_lam_percentile_zero() -> None:
    """Test _create_cell_pix_from_masks with lam_percentile=0."""
    cell_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    cell_pix = _create_cell_pix_from_masks([cell_mask], 3, 3, lam_percentile=0.0)

    # With lam_percentile=0, cell_core should just be where lammap > 0
    assert cell_pix.shape == (3, 3)
    assert cell_pix[1, 1] > 0  # Center pixel should be marked


def test_create_neuropil_from_dilation_single_cell() -> None:
    """Test create_neuropil_from_dilation with a single cell."""
    # Create a small cell in the center
    height, width = 20, 20
    cell_mask = np.zeros((height, width), dtype=bool)
    cell_mask[8:12, 8:12] = True  # 4x4 cell

    cell_masks = [cell_mask]

    eroded, neuropil = create_neuropil_from_dilation(
        cell_masks,
        height,
        width,
        inner_neuropil_radius=2,
        min_neuropil_pixels=50,
    )

    assert len(eroded) == 1
    assert len(neuropil) == 1

    # Eroded mask should be smaller than original
    assert np.sum(eroded[0]) <= np.sum(cell_mask)

    # Neuropil should not overlap with original cell
    assert not np.any(eroded[0] & neuropil[0])

    # Neuropil should have pixels
    assert np.sum(neuropil[0]) > 0


def test_create_neuropil_from_dilation_empty_cell() -> None:
    """Test create_neuropil_from_dilation with an empty cell."""
    height, width = 10, 10
    empty_mask = np.zeros((height, width), dtype=bool)
    cell_masks = [empty_mask]

    eroded, neuropil = create_neuropil_from_dilation(cell_masks, height, width)

    assert len(eroded) == 1
    assert len(neuropil) == 1

    # Both should be empty
    assert np.sum(eroded[0]) == 0
    assert np.sum(neuropil[0]) == 0


def test_create_neuropil_from_dilation_multiple_cells() -> None:
    """Test create_neuropil_from_dilation with multiple cells."""
    height, width = 30, 30

    # Create two cells far apart
    cell1 = np.zeros((height, width), dtype=bool)
    cell1[5:10, 5:10] = True

    cell2 = np.zeros((height, width), dtype=bool)
    cell2[20:25, 20:25] = True

    cell_masks = [cell1, cell2]

    eroded, neuropil = create_neuropil_from_dilation(
        cell_masks,
        height,
        width,
        inner_neuropil_radius=2,
        min_neuropil_pixels=30,
    )

    assert len(eroded) == 2
    assert len(neuropil) == 2

    # Each neuropil should not overlap with its corresponding eroded cell
    for i in range(2):
        assert not np.any(eroded[i] & neuropil[i])
        # Neuropil should have pixels
        assert np.sum(neuropil[i]) > 0


def test_create_neuropil_from_dilation_min_pixels() -> None:
    """Test that min_neuropil_pixels parameter affects the result."""
    height, width = 50, 50
    cell_mask = np.zeros((height, width), dtype=bool)
    cell_mask[20:25, 20:25] = True

    # Create neuropil with small min_pixels
    _, neuropil_small = create_neuropil_from_dilation(
        [cell_mask], height, width, min_neuropil_pixels=50
    )

    # Create neuropil with larger min_pixels
    _, neuropil_large = create_neuropil_from_dilation(
        [cell_mask], height, width, min_neuropil_pixels=200
    )

    # Larger min_pixels should result in more neuropil pixels
    assert np.sum(neuropil_large[0]) >= np.sum(neuropil_small[0])


def test_create_neuropil_from_dilation_inner_radius() -> None:
    """Test that inner_neuropil_radius creates a gap between cell and neuropil."""
    height, width = 30, 30
    cell_mask = np.zeros((height, width), dtype=bool)
    cell_mask[10:15, 10:15] = True

    _, neuropil = create_neuropil_from_dilation(
        [cell_mask],
        height,
        width,
        inner_neuropil_radius=3,
        min_neuropil_pixels=30,
    )

    # Create a dilated version of the original cell to check the gap
    from scipy.ndimage import binary_dilation

    dilated_cell = binary_dilation(cell_mask, iterations=3)

    # Neuropil should not overlap with the dilated cell (the forbidden zone)
    overlap = np.sum(neuropil[0] & dilated_cell)

    # There should be minimal to no overlap (accounting for edge effects)
    # The neuropil starts BEYOND the inner_neuropil_radius
    assert overlap == 0 or overlap < 5  # Allow small edge effects


def test_create_neuropil_respects_other_cells() -> None:
    """Test that neuropil masks exclude pixels occupied by other cells."""
    height, width = 20, 20

    # Create two cells close together
    cell1 = np.zeros((height, width), dtype=bool)
    cell1[5:10, 5:10] = True

    cell2 = np.zeros((height, width), dtype=bool)
    cell2[5:10, 11:16] = True  # Right next to cell1

    cell_masks = [cell1, cell2]

    _, neuropil = create_neuropil_from_dilation(
        cell_masks,
        height,
        width,
        inner_neuropil_radius=1,
        min_neuropil_pixels=20,
    )

    # Neuropil of cell1 should not significantly overlap with cell2
    overlap1_with_cell2 = np.sum(neuropil[0] & cell2)

    # Neuropil of cell2 should not significantly overlap with cell1
    overlap2_with_cell1 = np.sum(neuropil[1] & cell1)

    # Overlaps should be zero or very small (due to the cell_pix logic)
    assert overlap1_with_cell2 < 3
    assert overlap2_with_cell1 < 3
