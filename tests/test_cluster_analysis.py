"""Tests for cluster analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from cali.analysis._cluster_analysis import (
    ClusterResult,
    _find_optimal_k,
    _run_clustering,
    compute_cluster_analysis,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block_corr_matrix(n_per_cluster: int, n_clusters: int) -> np.ndarray:
    """Create a synthetic block-diagonal correlation matrix."""
    n = n_per_cluster * n_clusters
    corr = np.eye(n)
    for c in range(n_clusters):
        start = c * n_per_cluster
        end = start + n_per_cluster
        corr[start:end, start:end] = 0.8
    np.fill_diagonal(corr, 1.0)
    # Add small noise
    noise = np.random.default_rng(42).normal(0, 0.02, (n, n))
    corr = corr + noise
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1, 1)


# ---------------------------------------------------------------------------
# compute_cluster_analysis
# ---------------------------------------------------------------------------


def test_returns_none_for_too_few_rois() -> None:
    """Returns None when the correlation matrix has fewer than 3 ROIs."""
    corr = np.eye(2)
    assert compute_cluster_analysis(corr) is None


def test_returns_none_for_single_roi() -> None:
    """Returns None for a 1x1 correlation matrix."""
    corr = np.eye(1)
    assert compute_cluster_analysis(corr) is None


def test_hierarchical_fixed_k() -> None:
    """Returns a valid ClusterResult when k is fixed."""
    corr = _make_block_corr_matrix(5, 3)  # 15 ROIs, 3 clusters
    result = compute_cluster_analysis(corr, method="hierarchical", n_clusters=3)

    assert result is not None
    assert isinstance(result, ClusterResult)
    assert result.n_clusters == 3
    assert len(result.labels) == 15
    assert set(result.labels) == {0, 1, 2}
    assert -1.0 <= result.silhouette_score <= 1.0
    assert len(result.order) == 15


def test_auto_detect_k() -> None:
    """Auto-detection picks a valid k from the block structure."""
    corr = _make_block_corr_matrix(5, 3)
    result = compute_cluster_analysis(
        corr, method="hierarchical", n_clusters=0, max_k=10
    )

    assert result is not None
    assert 2 <= result.n_clusters <= 10
    assert result.silhouette_score > 0  # block structure clusters well


def test_order_groups_clusters() -> None:
    """order is a stable argsort by cluster label (non-decreasing)."""
    corr = _make_block_corr_matrix(5, 3)
    result = compute_cluster_analysis(corr, method="hierarchical", n_clusters=3)

    assert result is not None
    ordered_labels = [result.labels[i] for i in result.order]
    assert ordered_labels == sorted(ordered_labels)


def test_silhouette_in_range() -> None:
    """Silhouette score is always in [-1, 1]."""
    corr = _make_block_corr_matrix(10, 2)
    result = compute_cluster_analysis(corr, method="hierarchical", n_clusters=2)

    assert result is not None
    assert -1.0 <= result.silhouette_score <= 1.0


def test_returns_none_when_k_forced_below_two() -> None:
    """Returns None when a forced n_clusters clamps to k < 2 (n_clusters=1)."""
    # With n_rois=3: k = min(1, n_rois-1=2) = 1 < 2 → None
    corr = _make_block_corr_matrix(1, 3)  # 3 ROIs
    result = compute_cluster_analysis(corr, n_clusters=1)
    assert result is None


@pytest.mark.parametrize(
    ("n_per_cluster", "n_clusters", "fixed_k"),
    [
        (4, 2, 2),
        (5, 3, 3),
        (3, 4, 4),
    ],
)
def test_fixed_k_parametrized(
    n_per_cluster: int, n_clusters: int, fixed_k: int
) -> None:
    """Fixed k produces a result with exactly that many clusters."""
    corr = _make_block_corr_matrix(n_per_cluster, n_clusters)
    result = compute_cluster_analysis(corr, n_clusters=fixed_k)

    assert result is not None
    assert result.n_clusters == fixed_k
    assert len(result.labels) == n_per_cluster * n_clusters
    assert len(result.order) == n_per_cluster * n_clusters


def test_n_clusters_larger_than_n_rois_minus_one_clamped() -> None:
    """When n_clusters > n_rois - 1, clusters are clamped and result is valid."""
    # 4 ROIs, request 100 clusters → clamped to min(100, 3) = 3
    corr = _make_block_corr_matrix(2, 2)  # 4 ROIs
    result = compute_cluster_analysis(corr, n_clusters=100)

    assert result is not None
    assert result.n_clusters == 3  # min(100, 4-1)


# ---------------------------------------------------------------------------
# _find_optimal_k
# ---------------------------------------------------------------------------


def test_find_optimal_k_returns_two_when_k_max_below_two() -> None:
    """Returns 2 when max_k forces k_max < 2 (tiny matrix or max_k=1)."""
    dist = np.zeros((3, 3))
    # k_max = min(1, 2) = 1 < 2 → returns default of 2
    k = _find_optimal_k(dist, max_k=1)
    assert k == 2


def test_find_optimal_k_on_block_matrix() -> None:
    """Finds k > 1 on a clearly structured distance matrix."""
    corr = _make_block_corr_matrix(5, 3)
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    k = _find_optimal_k(dist, max_k=10)
    assert 2 <= k <= 10


# ---------------------------------------------------------------------------
# _run_clustering
# ---------------------------------------------------------------------------


def test_run_clustering_returns_zero_indexed_labels() -> None:
    """Cluster labels are 0-indexed (fcluster is 1-indexed, we subtract 1)."""
    corr = _make_block_corr_matrix(4, 2)
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    labels = _run_clustering(dist, k=2)

    assert labels.min() == 0
    assert labels.max() == 1
    assert len(labels) == 8


def test_run_clustering_produces_correct_number_of_clusters() -> None:
    """_run_clustering produces exactly k unique labels."""
    corr = _make_block_corr_matrix(3, 3)
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    labels = _run_clustering(dist, k=3)

    assert len(set(labels)) == 3
