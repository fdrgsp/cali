"""Cluster analysis for grouping ROIs by functional similarity.

Provides hierarchical (average/UPGMA linkage) clustering on correlation matrices,
with automatic optimal-k selection via silhouette score.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from cali._constants import CLUSTER_METHOD_HIERARCHICAL
from cali.logger import cali_logger


class ClusterResult(NamedTuple):
    """Result of cluster analysis on an FOV."""

    labels: list[int]
    n_clusters: int
    silhouette_score: float
    order: list[int]


def compute_cluster_analysis(
    corr_matrix: np.ndarray,
    method: str = CLUSTER_METHOD_HIERARCHICAL,
    n_clusters: int = 0,
    max_k: int = 10,
) -> ClusterResult | None:
    """Cluster ROIs based on their pairwise correlation matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray
        NxN symmetric correlation matrix (values in [-1, 1]).
        Matrix[i,j] = Pearson correlation between ROI i and ROI j.
    method : str
        Clustering method. Only "hierarchical" (average/UPGMA linkage) is supported.
    n_clusters : int
        Number of clusters. 0 = auto-detect via silhouette score.
    max_k : int
        Maximum k to test during auto-detection.

    Returns
    -------
    ClusterResult | None
        Clustering results, or None if clustering failed (e.g., too few ROIs).
    """
    n_rois = corr_matrix.shape[0]

    if n_rois < 3:
        cali_logger.info(
            f"Cluster analysis requires >= 3 ROIs, got {n_rois}. Skipping."
        )
        return None

    # Convert correlation to distance: d = 1 - r
    # Clip to [0, 2] to handle floating point issues
    dist_matrix = np.clip(1.0 - corr_matrix, 0.0, 2.0)
    # Zero out diagonal (self-distance = 0)
    np.fill_diagonal(dist_matrix, 0.0)

    # Determine k
    if n_clusters <= 0:
        k = _find_optimal_k(dist_matrix, max_k)
    else:
        k = min(n_clusters, n_rois - 1)

    if k < 2:
        cali_logger.info("Could not determine valid k for clustering. Skipping.")
        return None

    # Run clustering
    labels = _run_clustering(dist_matrix, k)

    # silhouette_score requires at least 2 unique labels
    if len(set(labels)) < 2:
        cali_logger.info(
            "Clustering produced only 1 unique label (degenerate correlation matrix). "
            "Skipping."
        )
        return None

    # Compute silhouette score
    sil_score = float(silhouette_score(dist_matrix, labels, metric="precomputed"))

    # Compute display order: sort ROI indices by cluster label
    order = [int(i) for i in np.argsort(labels, kind="stable")]

    cali_logger.info(
        f"Cluster analysis: method={method}, k={k}, silhouette={sil_score:.3f}"
    )

    return ClusterResult(
        labels=[int(lbl) for lbl in labels],
        n_clusters=k,
        silhouette_score=sil_score,
        order=order,
    )


def _find_optimal_k(dist_matrix: np.ndarray, max_k: int) -> int:
    """Find optimal number of clusters using silhouette score.

    Scans k from 2 to min(max_k, n_rois - 1) and returns the k
    with the highest silhouette score.

    Parameters
    ----------
    dist_matrix : np.ndarray
        NxN distance matrix (1 - correlation).
    max_k : int
        Maximum k to test.

    Returns
    -------
    int
        Optimal number of clusters. Returns 2 if scan fails.
    """
    n_rois = dist_matrix.shape[0]
    k_max = min(max_k, n_rois - 1)

    if k_max < 2:
        return 2

    best_k = 2
    best_score = -1.0

    for k in range(2, k_max + 1):
        labels = _run_clustering(dist_matrix, k)

        # silhouette_score requires at least 2 unique labels
        if len(set(labels)) < 2:  # pragma: no cover
            continue

        score = float(silhouette_score(dist_matrix, labels, metric="precomputed"))

        if score > best_score:
            best_score = score
            best_k = k

    cali_logger.info(f"Auto-detected optimal k={best_k} (silhouette={best_score:.3f})")
    return best_k


def _run_clustering(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """Run hierarchical (average/UPGMA linkage) clustering and return labels.

    Parameters
    ----------
    dist_matrix : np.ndarray
        NxN distance matrix (1 - correlation).
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Array of cluster labels (0-indexed), length N.
    """
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    # fcluster returns 1-indexed labels; convert to 0-indexed
    return fcluster(Z, t=k, criterion="maxclust") - 1
