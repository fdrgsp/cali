"""Test for plate_map_hash functionality."""

from cali.sqlmodel._model import CaliResult
from cali.sqlmodel._plate_map_util import compute_plate_map_hash


def test_compute_plate_map_hash_none() -> None:
    """Test that compute_plate_map_hash returns None for None input."""
    assert compute_plate_map_hash(None) is None


def test_compute_plate_map_hash_stable() -> None:
    """Test that same plate_maps produce same hash."""
    plate_maps1 = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle", "A2": "Drug"},
    }
    plate_maps2 = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle", "A2": "Drug"},
    }

    hash1 = compute_plate_map_hash(plate_maps1)
    hash2 = compute_plate_map_hash(plate_maps2)

    assert hash1 == hash2
    assert hash1 is not None


def test_compute_plate_map_hash_order_independent() -> None:
    """Test that hash is independent of dict iteration order."""
    # Different insertion order
    plate_maps1 = {
        "genotype": {"A1": "WT", "A2": "KO"},
        "treatment": {"A1": "Vehicle"},
    }
    plate_maps2 = {
        "treatment": {"A1": "Vehicle"},
        "genotype": {"A2": "KO", "A1": "WT"},  # Different well order too
    }

    hash1 = compute_plate_map_hash(plate_maps1)
    hash2 = compute_plate_map_hash(plate_maps2)

    assert hash1 == hash2


def test_compute_plate_map_hash_different_values() -> None:
    """Test that different plate_maps produce different hashes."""
    plate_maps1 = {
        "genotype": {"A1": "WT", "A2": "KO"},
    }
    plate_maps2 = {
        "genotype": {"A1": "KO", "A2": "WT"},  # Swapped values
    }

    hash1 = compute_plate_map_hash(plate_maps1)
    hash2 = compute_plate_map_hash(plate_maps2)

    assert hash1 != hash2


def test_cali_result_equality_with_plate_map_hash() -> None:
    """Test CaliResult equality considers plate_map_hash."""
    result1 = CaliResult(
        experiment=1,
        detection_settings=1,
        extraction_settings_id=1,
        analysis_settings_id=1,
        plate_map_hash="abc123",
        positions_analyzed=[0, 1],
    )

    result2 = CaliResult(
        experiment=1,
        detection_settings=1,
        extraction_settings_id=1,
        analysis_settings_id=1,
        plate_map_hash="abc123",  # Same hash
        positions_analyzed=[0, 1],
    )

    result3 = CaliResult(
        experiment=1,
        detection_settings=1,
        extraction_settings_id=1,
        analysis_settings_id=1,
        plate_map_hash="xyz789",  # Different hash
        positions_analyzed=[0, 1],
    )

    # Same plate_map_hash -> equal
    assert result1 == result2
    assert hash(result1) == hash(result2)

    # Different plate_map_hash -> not equal
    assert result1 != result3
    assert hash(result1) != hash(result3)


def test_cali_result_equality_none_plate_map_hash() -> None:
    """Test CaliResult equality when plate_map_hash is None."""
    result1 = CaliResult(
        experiment=1,
        detection_settings=1,
        plate_map_hash=None,
        positions_detected=[0],
    )

    result2 = CaliResult(
        experiment=1,
        detection_settings=1,
        plate_map_hash=None,
        positions_detected=[0],
    )

    result3 = CaliResult(
        experiment=1,
        detection_settings=1,
        plate_map_hash="abc123",
        positions_detected=[0],
    )

    # Both None -> equal
    assert result1 == result2
    assert hash(result1) == hash(result2)

    # One None, one not -> not equal
    assert result1 != result3
    assert hash(result1) != hash(result3)
