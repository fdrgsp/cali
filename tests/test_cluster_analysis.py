"""Tests for cluster analysis module."""

import numpy as np

from cali.analysis._cluster_analysis import (
    ClusterResult,
    compute_cluster_analysis,
)


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


class TestComputeClusterAnalysis:
    def test_returns_none_for_too_few_rois(self) -> None:
        corr = np.eye(2)
        result = compute_cluster_analysis(corr)
        assert result is None

    def test_hierarchical_fixed_k(self) -> None:
        corr = _make_block_corr_matrix(5, 3)  # 15 ROIs, 3 clusters
        result = compute_cluster_analysis(corr, method="hierarchical", n_clusters=3)
        assert result is not None
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 3
        assert len(result.labels) == 15
        assert set(result.labels) == {0, 1, 2}
        assert -1 <= result.silhouette_score <= 1
        assert len(result.order) == 15

    def test_auto_detect_k(self) -> None:
        corr = _make_block_corr_matrix(5, 3)
        result = compute_cluster_analysis(
            corr, method="hierarchical", n_clusters=0, max_k=10
        )
        assert result is not None
        assert 2 <= result.n_clusters <= 10
        assert result.silhouette_score > 0  # block structure should cluster well

    def test_order_groups_clusters(self) -> None:
        corr = _make_block_corr_matrix(5, 3)
        result = compute_cluster_analysis(corr, method="hierarchical", n_clusters=3)
        assert result is not None
        # Verify that order sorts by cluster
        ordered_labels = [result.labels[i] for i in result.order]
        # Should be non-decreasing
        assert ordered_labels == sorted(ordered_labels)

    def test_silhouette_in_range(self) -> None:
        corr = _make_block_corr_matrix(10, 2)
        result = compute_cluster_analysis(corr, method="hierarchical", n_clusters=2)
        assert result is not None
        assert -1.0 <= result.silhouette_score <= 1.0
