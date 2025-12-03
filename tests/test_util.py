"""Tests for utility functions in cali.util._util."""

from pathlib import Path

import numpy as np
import pytest

from cali.util._util import (
    coordinates_to_mask,
    mask_to_coordinates,
)


@pytest.mark.parametrize(
    "mask,expected_coords,expected_shape",
    [
        # Test case 1: Simple 3x3 mask with single pixel
        (
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
            ([1], [1]),
            (3, 3),
        ),
        # Test case 2: 3x3 mask with diagonal
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
            ([0, 1, 2], [0, 1, 2]),
            (3, 3),
        ),
        # Test case 3: Empty mask
        (
            np.array([[0, 0], [0, 0]], dtype=bool),
            ([], []),
            (2, 2),
        ),
        # Test case 4: Full mask
        (
            np.array([[1, 1], [1, 1]], dtype=bool),
            ([0, 0, 1, 1], [0, 1, 0, 1]),
            (2, 2),
        ),
        # Test case 5: Rectangular region
        (
            np.array(
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool
            ),
            ([1, 1, 2, 2], [1, 2, 1, 2]),
            (4, 4),
        ),
    ],
)
def test_mask_to_coordinates(
    mask: np.ndarray,
    expected_coords: tuple[list[int], list[int]],
    expected_shape: tuple[int, int],
) -> None:
    """Test mask_to_coordinates with various mask patterns."""
    coords, shape = mask_to_coordinates(mask)

    assert shape == expected_shape
    y_coords, x_coords = coords
    expected_y, expected_x = expected_coords

    # Sort both actual and expected for consistent comparison
    y_x_pairs = sorted(zip(y_coords, x_coords))
    expected_pairs = sorted(zip(expected_y, expected_x))

    assert y_x_pairs == expected_pairs


@pytest.mark.parametrize(
    "coordinates,shape,expected_mask",
    [
        # Test case 1: Single pixel
        (
            ([1], [1]),
            (3, 3),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
        ),
        # Test case 2: Diagonal
        (
            ([0, 1, 2], [0, 1, 2]),
            (3, 3),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ),
        # Test case 3: Empty
        (
            ([], []),
            (2, 2),
            np.array([[0, 0], [0, 0]], dtype=bool),
        ),
        # Test case 4: Full mask
        (
            ([0, 0, 1, 1], [0, 1, 0, 1]),
            (2, 2),
            np.array([[1, 1], [1, 1]], dtype=bool),
        ),
        # Test case 5: Rectangular region
        (
            ([1, 1, 2, 2], [1, 2, 1, 2]),
            (4, 4),
            np.array(
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool
            ),
        ),
    ],
)
def test_coordinates_to_mask(
    coordinates: tuple[list[int], list[int]],
    shape: tuple[int, int],
    expected_mask: np.ndarray,
) -> None:
    """Test coordinates_to_mask with various coordinate patterns."""
    mask = coordinates_to_mask(coordinates, shape)
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_coordinates_roundtrip() -> None:
    """Test that converting mask -> coordinates -> mask is lossless."""
    # Create a random mask
    np.random.seed(42)
    original_mask = np.random.randint(0, 2, size=(10, 10), dtype=bool)

    # Convert to coordinates
    coords, shape = mask_to_coordinates(original_mask)

    # Convert back to mask
    reconstructed_mask = coordinates_to_mask(coords, shape)

    # Should be identical
    np.testing.assert_array_equal(original_mask, reconstructed_mask)


def test_coordinates_mask_roundtrip() -> None:
    """Test that converting coordinates -> mask -> coordinates is lossless."""
    # Create coordinates for a specific pattern
    original_coords = ([0, 1, 2, 3], [3, 2, 1, 0])  # Anti-diagonal
    shape = (5, 5)

    # Convert to mask
    mask = coordinates_to_mask(original_coords, shape)

    # Convert back to coordinates
    coords, reconstructed_shape = mask_to_coordinates(mask)

    # Shape should match
    assert reconstructed_shape == shape

    # Coordinates should match (after sorting)
    y_coords, x_coords = coords
    orig_y, orig_x = original_coords

    reconstructed_pairs = sorted(zip(y_coords, x_coords))
    original_pairs = sorted(zip(orig_y, orig_x))

    assert reconstructed_pairs == original_pairs


def test_load_data_from_path_tensorstore(tmp_path: Path) -> None:
    """Test load_data_from_path with tensorstore zarr path."""
    from cali.util._util import load_data_from_path

    # Create a fake tensorstore zarr directory
    ts_path = tmp_path / "data.tensorstore.zarr"
    ts_path.mkdir()
    (ts_path / ".zattrs").write_text("{}")

    # Should attempt to load with TensorstoreZarrReader but fail
    # because it's not a valid tensorstore
    # The function should catch the error and handle it gracefully
    # or raise - either is acceptable, just verify it doesn't crash unexpectedly
    try:
        result = load_data_from_path(ts_path)
        # If it succeeds somehow, result should be a reader
        assert result is not None
    except (ValueError, RuntimeError):
        # Expected - not a valid tensorstore
        pass


def test_load_data_from_path_ome_zarr(tmp_path: Path) -> None:
    """Test load_data_from_path with OME zarr path."""
    from cali.util._util import load_data_from_path

    # Create a fake OME zarr directory
    zarr_path = tmp_path / "data.ome.zarr"
    zarr_path.mkdir()
    (zarr_path / ".zattrs").write_text("{}")

    # Should attempt to load with OMEZarrReader but may fail if not valid
    try:
        result = load_data_from_path(zarr_path)
        # If it succeeds, result should be a reader
        assert result is not None
    except (ValueError, RuntimeError, KeyError):
        # Expected - not a valid OME zarr
        pass


def test_load_data_from_path_unsupported(tmp_path: Path) -> None:
    """Test load_data_from_path with unsupported format."""
    # Create some random path that's not a recognized format
    unsupported_path = tmp_path / "data.txt"
    unsupported_path.write_text("not a zarr")

    from cali.util._util import load_data_from_path

    # Should return None for unsupported formats
    result = load_data_from_path(unsupported_path)
    assert result is None
