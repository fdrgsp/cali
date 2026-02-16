# Cluster Analysis Implementation Plan

## Overview

Add ROI clustering to the cali analysis pipeline. Two methods: **Hierarchical (Ward's linkage)** and **K-means**, operating on the existing `calcium_den_dff_corr_matrix` (denoised ΔF/F Pearson correlation matrix, already computed per-FOV in `FOVAnalysis`).

Automatic optimal-k selection via silhouette score, with user override. Results stored in `FOVAnalysis` table. Three new visualization plots in a "Cluster Analysis" category.

---

## Files to Modify (in order)

| # | File | Action | Description |
|---|------|--------|-------------|
| 1 | `pyproject.toml` | MODIFY | Add `scikit-learn` dependency |
| 2 | `src/cali/_constants.py` | MODIFY | Add default constants for clustering |
| 3 | `src/cali/sqlmodel/_model.py` | MODIFY | Add cluster fields to `AnalysisSettings` + `FOVAnalysis` |
| 4 | `src/cali/analysis/_cluster_analysis.py` | **NEW** | Core clustering algorithms |
| 5 | `src/cali/analysis/_fov_analysis.py` | MODIFY | Call clustering after correlation |
| 6 | `src/cali/analysis/_fov_analysis_parallel.py` | MODIFY | Same integration for parallel path |
| 7 | `src/cali/gui/_analysis_gui.py` | MODIFY | Add cluster settings to GUI |
| 8 | `src/cali/plot/_single_wells_plots/cluster/__init__.py` | **NEW** | Empty init |
| 9 | `src/cali/plot/_single_wells_plots/cluster/_plot_cluster_analysis.py` | **NEW** | 3 plot functions |
| 10 | `src/cali/plot/_main_plot.py` | MODIFY | Register 3 new plot products |

---

## Step 1: Add `scikit-learn` dependency

**File:** `pyproject.toml` (line ~49)

Add `"scikit-learn"` to the `dependencies` list, after `"scipy"`:

```python
# Current:
    "scipy",
# Change to:
    "scipy",
    "scikit-learn",
```

---

## Step 2: Add default constants

**File:** `src/cali/_constants.py`

Add after the existing defaults (around line 99):

```python
# Cluster analysis defaults
DEFAULT_CLUSTER_METHOD = "hierarchical"  # "hierarchical" or "kmeans"
DEFAULT_CLUSTER_N_CLUSTERS = 0  # 0 = auto-detect via silhouette score
DEFAULT_CLUSTER_MAX_K = 10  # max k to scan during auto-detection
CLUSTER_METHOD_HIERARCHICAL = "hierarchical"
CLUSTER_METHOD_KMEANS = "kmeans"
```

---

## Step 3: Add cluster fields to database models

**File:** `src/cali/sqlmodel/_model.py`

### 3a. Add to `AnalysisSettings` class (after `enable_rising_edge_analysis` field, ~line 1152)

```python
    # Cluster analysis settings
    cluster_method: str = DEFAULT_CLUSTER_METHOD
    cluster_n_clusters: int = DEFAULT_CLUSTER_N_CLUSTERS
    cluster_max_k: int = DEFAULT_CLUSTER_MAX_K
```

Import the new constants at the top of the file from `cali._constants`.

### 3b. Add to `FOVAnalysis` class (after the last `calcium_population_activity_raw` field, ~line 1860)

```python
    # Cluster analysis results
    cluster_labels: list[int] | None = Field(default=None, sa_column=Column(JSON))
    cluster_method: str | None = Field(default=None)
    cluster_n_clusters: int | None = Field(default=None)
    cluster_silhouette_score: float | None = Field(default=None)
    cluster_order: list[int] | None = Field(default=None, sa_column=Column(JSON))
```

**Field meanings:**
- `cluster_labels`: Cluster assignment per ROI, same order as `active_roi_labels`. e.g. `[0, 2, 1, 0, 1, ...]`
- `cluster_method`: `"hierarchical"` or `"kmeans"` - which method produced these results
- `cluster_n_clusters`: The final k used (may differ from settings if auto-detected)
- `cluster_silhouette_score`: Silhouette score of the clustering result ([-1, 1], higher = better)
- `cluster_order`: Indices into `active_roi_labels` sorted by cluster assignment, for display. e.g. `[0, 3, 2, 4, 1, ...]` means show ROI at position 0 first, then 3, etc.

---

## Step 4: Create clustering logic module

**New file:** `src/cali/analysis/_cluster_analysis.py`

```python
"""Cluster analysis for grouping ROIs by functional similarity.

Provides hierarchical (Ward's linkage) and K-means clustering on
correlation matrices, with automatic optimal-k selection via silhouette score.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cali._constants import CLUSTER_METHOD_HIERARCHICAL, CLUSTER_METHOD_KMEANS
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
        Clustering method: "hierarchical" (Ward's linkage) or "kmeans".
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
        k = _find_optimal_k(dist_matrix, corr_matrix, method, max_k)
    else:
        k = min(n_clusters, n_rois - 1)

    if k < 2:
        cali_logger.info("Could not determine valid k for clustering. Skipping.")
        return None

    # Run clustering
    labels = _run_clustering(dist_matrix, corr_matrix, method, k)

    # Compute silhouette score
    sil_score = float(
        silhouette_score(dist_matrix, labels, metric="precomputed")
    )

    # Compute display order: sort ROI indices by cluster label
    order = list(np.argsort(labels))

    cali_logger.info(
        f"Cluster analysis: method={method}, k={k}, "
        f"silhouette={sil_score:.3f}"
    )

    return ClusterResult(
        labels=[int(lbl) for lbl in labels],
        n_clusters=k,
        silhouette_score=sil_score,
        order=order,
    )


def _find_optimal_k(
    dist_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    method: str,
    max_k: int,
) -> int:
    """Find optimal number of clusters using silhouette score.

    Scans k from 2 to min(max_k, n_rois - 1) and returns the k
    with the highest silhouette score.

    Parameters
    ----------
    dist_matrix : np.ndarray
        NxN distance matrix (1 - correlation).
    corr_matrix : np.ndarray
        NxN correlation matrix (used as features for K-means).
    method : str
        Clustering method to use for evaluation.
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
        labels = _run_clustering(dist_matrix, corr_matrix, method, k)

        # silhouette_score requires at least 2 unique labels
        if len(set(labels)) < 2:
            continue

        score = float(
            silhouette_score(dist_matrix, labels, metric="precomputed")
        )

        if score > best_score:
            best_score = score
            best_k = k

    cali_logger.info(
        f"Auto-detected optimal k={best_k} (silhouette={best_score:.3f})"
    )
    return best_k


def _run_clustering(
    dist_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    method: str,
    k: int,
) -> np.ndarray:
    """Run clustering with the specified method and return labels.

    Parameters
    ----------
    dist_matrix : np.ndarray
        NxN distance matrix (1 - correlation). Used for hierarchical clustering
        and silhouette evaluation.
    corr_matrix : np.ndarray
        NxN correlation matrix. Used as feature matrix for K-means
        (each ROI's row = its correlation profile with all other ROIs).
    method : str
        "hierarchical" or "kmeans".
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Array of cluster labels (0-indexed), length N.
    """
    if method == CLUSTER_METHOD_HIERARCHICAL:
        # Ward's linkage on condensed distance matrix
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="ward")
        # fcluster returns 1-indexed labels; convert to 0-indexed
        labels = fcluster(Z, t=k, criterion="maxclust") - 1
    elif method == CLUSTER_METHOD_KMEANS:
        # K-means on correlation features (each ROI = row of corr_matrix)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(corr_matrix)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}")

    return labels
```

---

## Step 5: Integrate into `_fov_analysis.py`

**File:** `src/cali/analysis/_fov_analysis.py`

### 5a. Add import at top (after existing imports):

```python
from cali.analysis._cluster_analysis import compute_cluster_analysis
```

### 5b. Add clustering computation after line 181 (after `calcium_den_dff_corr_matrix` is computed):

```python
    # Cluster analysis on denoised ΔF/F correlation matrix
    cluster_labels = None
    cluster_method_used = None
    cluster_n = None
    cluster_silhouette = None
    cluster_order = None

    if calcium_den_dff_corr_matrix is not None and len(roi_labels) >= 3:
        cluster_result = compute_cluster_analysis(
            corr_matrix=calcium_den_dff_corr_matrix,
            method=analysis_settings.cluster_method,
            n_clusters=analysis_settings.cluster_n_clusters,
            max_k=analysis_settings.cluster_max_k,
        )
        if cluster_result is not None:
            cluster_labels = cluster_result.labels
            cluster_method_used = analysis_settings.cluster_method
            cluster_n = cluster_result.n_clusters
            cluster_silhouette = cluster_result.silhouette_score
            cluster_order = cluster_result.order
```

### 5c. Add to `FOVAnalysis(...)` constructor (at the end, before the closing paren, around line 438):

```python
        # Cluster analysis results
        cluster_labels=cluster_labels,
        cluster_method=cluster_method_used,
        cluster_n_clusters=cluster_n,
        cluster_silhouette_score=cluster_silhouette,
        cluster_order=cluster_order,
```

---

## Step 6: Integrate into `_fov_analysis_parallel.py`

**File:** `src/cali/analysis/_fov_analysis_parallel.py`

Same pattern as Step 5, but in `compute_fov_analysis_parallel()`:

### 6a. Add import at top:

```python
from cali.analysis._cluster_analysis import compute_cluster_analysis
```

### 6b. Add clustering after line 251 (after `calcium_den_dff_corr_matrix` is computed):

```python
    # Cluster analysis on denoised ΔF/F correlation matrix
    cluster_labels = None
    cluster_method_used = None
    cluster_n = None
    cluster_silhouette = None
    cluster_order = None

    if calcium_den_dff_corr_matrix is not None and len(roi_labels) >= 3:
        cluster_result = compute_cluster_analysis(
            corr_matrix=calcium_den_dff_corr_matrix,
            method=analysis_settings.cluster_method,
            n_clusters=analysis_settings.cluster_n_clusters,
            max_k=analysis_settings.cluster_max_k,
        )
        if cluster_result is not None:
            cluster_labels = cluster_result.labels
            cluster_method_used = analysis_settings.cluster_method
            cluster_n = cluster_result.n_clusters
            cluster_silhouette = cluster_result.silhouette_score
            cluster_order = cluster_result.order
```

### 6c. Add to `FOVAnalysis(...)` constructor (at end, before closing paren, ~line 557):

```python
        # Cluster analysis results
        cluster_labels=cluster_labels,
        cluster_method=cluster_method_used,
        cluster_n_clusters=cluster_n,
        cluster_silhouette_score=cluster_silhouette,
        cluster_order=cluster_order,
```

---

## Step 7: Add cluster settings to GUI

**File:** `src/cali/gui/_analysis_gui.py`

### 7a. Add imports at top (add to existing `_constants` import):

```python
from cali._constants import (
    # ... existing imports ...
    DEFAULT_CLUSTER_METHOD,
    DEFAULT_CLUSTER_N_CLUSTERS,
    DEFAULT_CLUSTER_MAX_K,
    CLUSTER_METHOD_HIERARCHICAL,
    CLUSTER_METHOD_KMEANS,
)
```

### 7b. Add `ClusterData` dataclass (after `SpikeData`, ~line 122):

```python
@dataclass(frozen=True)
class ClusterData:
    """Data structure to hold the cluster analysis settings."""

    cluster_method: str = DEFAULT_CLUSTER_METHOD
    cluster_n_clusters: int = DEFAULT_CLUSTER_N_CLUSTERS
    cluster_max_k: int = DEFAULT_CLUSTER_MAX_K
```

### 7c. Add `cluster_data` field to `AnalysisSettingsData` (after `spikes_data`):

```python
@dataclass(frozen=True)
class AnalysisSettingsData:
    """Data structure to hold the analysis settings."""

    calcium_peaks_data: CalciumPeaksData | None = None
    spikes_data: SpikeData | None = None
    cluster_data: ClusterData | None = None  # <-- NEW
    experiment_type_data: ExperimentTypeData | None = None
    frame_rate: float = DEFAULT_FRAME_RATE
    threads: int = max((os.cpu_count() or 1) - 2, 1)
    n_processes: int = max((os.cpu_count() or 1) - 2, 1)
    export_options: dict[str, tuple[bool, int, int]] | None = None
    export_enabled: bool = False
```

### 7d. Create `_ClusterWidget` class (add after `_BurstWidget` class, ~line 1030):

```python
class _ClusterWidget(QWidget):
    """Widget to configure cluster analysis settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Cluster Analysis Settings\n\n"
            "Groups ROIs into functional clusters based on their denoised ΔF/F "
            "correlation patterns.\n\n"
            "• Method: Hierarchical (Ward's linkage on correlation distance) "
            "or K-means (on correlation features).\n\n"
            "• Number of Clusters: Set to 0 for automatic detection via "
            "silhouette score analysis. Set to a positive integer to force "
            "that many clusters.\n\n"
            "• Max K (Auto): Upper bound for the silhouette scan when using "
            "auto-detection. Only relevant when 'Number of Clusters' is 0."
        )

        # Method selector
        self._method_lbl = QLabel("Cluster Method:", self)
        self._method_lbl.setSizePolicy(*FIXED)
        self._method_combo = QComboBox(self)
        self._method_combo.addItems(["Hierarchical (Ward)", "K-means"])

        # Number of clusters
        self._n_clusters_lbl = QLabel("Number of Clusters:", self)
        self._n_clusters_lbl.setSizePolicy(*FIXED)
        self._n_clusters_spin = QSpinBox(self)
        self._n_clusters_spin.setRange(0, 50)
        self._n_clusters_spin.setValue(DEFAULT_CLUSTER_N_CLUSTERS)
        self._n_clusters_spin.setSpecialValueText("Auto")

        # Max k for auto-detection
        self._max_k_lbl = QLabel("Max K (Auto):", self)
        self._max_k_lbl.setSizePolicy(*FIXED)
        self._max_k_spin = QSpinBox(self)
        self._max_k_spin.setRange(2, 50)
        self._max_k_spin.setValue(DEFAULT_CLUSTER_MAX_K)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._method_lbl, 0, 0)
        layout.addWidget(self._method_combo, 0, 1)
        layout.addWidget(self._n_clusters_lbl, 1, 0)
        layout.addWidget(self._n_clusters_spin, 1, 1)
        layout.addWidget(self._max_k_lbl, 2, 0)
        layout.addWidget(self._max_k_spin, 2, 1)

    def value(self) -> ClusterData:
        """Get the current cluster settings."""
        method = (
            CLUSTER_METHOD_HIERARCHICAL
            if self._method_combo.currentIndex() == 0
            else CLUSTER_METHOD_KMEANS
        )
        return ClusterData(
            cluster_method=method,
            cluster_n_clusters=self._n_clusters_spin.value(),
            cluster_max_k=self._max_k_spin.value(),
        )

    def setValue(self, value: ClusterData) -> None:
        """Set the cluster settings."""
        idx = 0 if value.cluster_method == CLUSTER_METHOD_HIERARCHICAL else 1
        self._method_combo.setCurrentIndex(idx)
        self._n_clusters_spin.setValue(value.cluster_n_clusters)
        self._max_k_spin.setValue(value.cluster_max_k)

    def reset(self) -> None:
        """Reset to defaults."""
        self._method_combo.setCurrentIndex(0)
        self._n_clusters_spin.setValue(DEFAULT_CLUSTER_N_CLUSTERS)
        self._max_k_spin.setValue(DEFAULT_CLUSTER_MAX_K)

    def set_labels_width(self, width: int) -> None:
        """Set fixed width for labels to align with other widgets."""
        self._method_lbl.setFixedWidth(width)
        self._n_clusters_lbl.setFixedWidth(width)
        self._max_k_lbl.setFixedWidth(width)
```

### 7e. Add `_ClusterWidget` to `_AnalysisGUI.__init__()` (~line 191):

After `self._spike_wdg = _SpikeWidget(self)`:
```python
        self._cluster_wdg = _ClusterWidget(self)
```

Add to layout after the "Inferred Spikes" section (~line 237):
```python
        group_layout.addWidget(create_divider_line("Cluster Analysis"))
        group_layout.addWidget(self._cluster_wdg)
```

Add label width alignment (~line 265):
```python
        self._cluster_wdg.set_labels_width(fix_width)
```

### 7f. Update `_AnalysisGUI.value()` method (~line 283):

```python
    def value(self) -> AnalysisSettingsData:
        """Get the current values of the widget."""
        return AnalysisSettingsData(
            calcium_peaks_data=self._calcium_peaks_wdg.value(),
            spikes_data=self._spike_wdg.value(),
            cluster_data=self._cluster_wdg.value(),  # <-- NEW
            experiment_type_data=self._experiment_type_wdg.value(),
            frame_rate=self._metadata_wdg.value(),
            threads=self._threads.value(),
            n_processes=self._n_processes.value(),
            export_options=self._export_group.get_options_for_data(),
            export_enabled=self._export_group.isChecked(),
        )
```

### 7g. Update `_AnalysisGUI.setValue()` method (~line 295):

Add after `self._spike_wdg.setValue(...)`:
```python
        if value.cluster_data is not None:
            self._cluster_wdg.setValue(value.cluster_data)
```

### 7h. Update `_AnalysisGUI.reset()` method (~line 311):

Add:
```python
        self._cluster_wdg.reset()
```

### 7i. Update `_AnalysisGUI.to_model_settings()` method (~line 345):

Add `cluster_data` extraction:
```python
        cluster_data = settings.cluster_data
```

Add to the `AnalysisSettings(...)` constructor call (after `enable_rising_edge_analysis`):
```python
            cluster_method=(
                cluster_data.cluster_method
                if cluster_data
                else DEFAULT_CLUSTER_METHOD
            ),
            cluster_n_clusters=(
                cluster_data.cluster_n_clusters
                if cluster_data
                else DEFAULT_CLUSTER_N_CLUSTERS
            ),
            cluster_max_k=(
                cluster_data.cluster_max_k
                if cluster_data
                else DEFAULT_CLUSTER_MAX_K
            ),
```

### 7j. Update `_cali_gui.py` where `AnalysisSettingsData` is restored from DB

In `_cali_gui.py`, wherever `AnalysisSettingsData(...)` is constructed from an `AnalysisSettings` loaded from the database, add `cluster_data`:

```python
                    cluster_data=ClusterData(
                        cluster_method=a_settings.cluster_method,
                        cluster_n_clusters=a_settings.cluster_n_clusters,
                        cluster_max_k=a_settings.cluster_max_k,
                    ),
```

Import `ClusterData` from `._analysis_gui`.

---

## Step 8: Create cluster plot module

### 8a. Create `__init__.py`

**New file:** `src/cali/plot/_single_wells_plots/cluster/__init__.py`

Empty file (just a blank file or with a module docstring).

### 8b. Create plot functions

**New file:** `src/cali/plot/_single_wells_plots/cluster/_plot_cluster_analysis.py`

```python
"""Cluster analysis visualization plots.

Provides three cluster visualization types:
1. Cluster-sorted correlation heatmap
2. Cluster-colored calcium peaks raster
3. Cluster-colored denoised ΔF/F traces
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import add_colorbar_to_widget, disconnect_hover_handlers
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, FOVAnalysis, Traces

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# ---- Color utilities ---- #

# Qualitative color palette (distinct, colorblind-friendly)
CLUSTER_COLORS = [
    (31, 119, 180, 255),   # blue
    (255, 127, 14, 255),   # orange
    (44, 160, 44, 255),    # green
    (214, 39, 40, 255),    # red
    (148, 103, 189, 255),  # purple
    (140, 86, 75, 255),    # brown
    (227, 119, 194, 255),  # pink
    (127, 127, 127, 255),  # gray
    (188, 189, 34, 255),   # olive
    (23, 190, 207, 255),   # cyan
]

CORR_CMAP_NAME = "viridis"
CORR_CMAP = pg.colormap.get(CORR_CMAP_NAME)


def _get_cluster_color(cluster_id: int) -> tuple[int, int, int, int]:
    """Get color for a cluster, cycling through palette if needed."""
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def _get_cluster_data_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None,
) -> tuple[
    np.ndarray | None,  # corr_matrix
    list[int] | None,   # roi_labels
    list[int] | None,   # cluster_labels
    list[int] | None,   # cluster_order
    str | None,         # cluster_method
    int | None,         # cluster_n_clusters
    float | None,       # cluster_silhouette_score
]:
    """Query FOVAnalysis for cluster data.

    Returns
    -------
    tuple
        (corr_matrix, roi_labels, cluster_labels, cluster_order,
         method, n_clusters, silhouette_score)
        All None if not found.
    """
    if run_id is None:
        return None, None, None, None, None, None, None

    try:
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .where(col(FOV.name) == fov_name)
                .where(col(FOVAnalysis.analysis_result_id) == run_id)
            )
            fov_analysis = session.exec(stmt).first()

            if fov_analysis is None:
                return None, None, None, None, None, None, None

            if (
                fov_analysis.cluster_labels is None
                or fov_analysis.active_roi_labels is None
            ):
                return None, None, None, None, None, None, None

            corr_matrix = (
                np.asarray(fov_analysis.calcium_den_dff_corr_matrix, dtype=float)
                if fov_analysis.calcium_den_dff_corr_matrix is not None
                else None
            )

            return (
                corr_matrix,
                list(fov_analysis.active_roi_labels),
                list(fov_analysis.cluster_labels),
                (
                    list(fov_analysis.cluster_order)
                    if fov_analysis.cluster_order is not None
                    else None
                ),
                fov_analysis.cluster_method,
                fov_analysis.cluster_n_clusters,
                fov_analysis.cluster_silhouette_score,
            )
    except Exception:
        cali_logger.exception("Error loading cluster data from database")
        return None, None, None, None, None, None, None


# ---- Plot 1: Cluster-Sorted Correlation Heatmap ---- #

def _plot_cluster_sorted_correlation_heatmap(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Plot correlation heatmap with rows/columns sorted by cluster.

    Shows the denoised ΔF/F correlation matrix reordered so ROIs in the
    same cluster are adjacent. Cluster boundaries drawn as white lines.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    (
        corr_matrix, roi_labels, cluster_labels, cluster_order,
        method, n_clusters, sil_score,
    ) = _get_cluster_data_from_db(engine, fov_name, run_id)

    if corr_matrix is None or cluster_labels is None or cluster_order is None:
        plot.setTitle("Cluster-Sorted Correlation (No cluster data)")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter by selected ROIs if needed
    if rois is not None:
        # Build index mapping for selected ROIs
        indices = [i for i, lbl in enumerate(roi_labels) if lbl in rois]
        if len(indices) < 3:
            plot.setTitle("Cluster-Sorted Correlation (Need ≥3 ROIs)")
            return
        # Re-sort by cluster within selected ROIs
        sub_labels = [cluster_labels[i] for i in indices]
        sorted_idx = sorted(range(len(indices)), key=lambda x: sub_labels[x])
        reorder = [indices[s] for s in sorted_idx]
        cluster_labels_ordered = [cluster_labels[i] for i in reorder]
    else:
        reorder = cluster_order
        cluster_labels_ordered = [cluster_labels[i] for i in reorder]

    # Reorder correlation matrix
    sorted_corr = corr_matrix[np.ix_(reorder, reorder)]

    # Display heatmap
    img = pg.ImageItem(sorted_corr)
    img.setLookupTable(CORR_CMAP.getLookupTable(0, 1, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb = plot.getViewBox()
    vb.invertY(True)
    vb.setAspectLocked(True)

    # Draw cluster boundary lines
    n = len(reorder)
    boundaries = []
    for i in range(1, n):
        if cluster_labels_ordered[i] != cluster_labels_ordered[i - 1]:
            boundaries.append(i)

    for b in boundaries:
        # Horizontal line
        h_line = pg.InfiniteLine(pos=b, angle=0, pen=pg.mkPen("w", width=2))
        plot.addItem(h_line)
        # Vertical line
        v_line = pg.InfiniteLine(pos=b, angle=90, pen=pg.mkPen("w", width=2))
        plot.addItem(v_line)

    method_str = method or "unknown"
    k_str = n_clusters or "?"
    sil_str = f"{sil_score:.3f}" if sil_score is not None else "N/A"
    plot.setTitle(
        f"Cluster-Sorted Correlation ({method_str}, k={k_str}, "
        f"silhouette={sil_str})"
    )
    plot.setLabel("bottom", "ROI (sorted by cluster)")
    plot.setLabel("left", "ROI (sorted by cluster)")
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CORR_CMAP_NAME
    )


# ---- Plot 2: Cluster-Colored Raster ---- #

def _plot_cluster_colored_raster(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Plot calcium peaks raster with events colored by cluster.

    Each ROI's peak events are plotted as scatter points, colored by
    cluster assignment. ROIs are sorted by cluster on the y-axis.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)
    vb.invertY(True)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get cluster data
    (
        _, roi_labels, cluster_labels, cluster_order,
        method, n_clusters, sil_score,
    ) = _get_cluster_data_from_db(engine, fov_name, run_id)

    if cluster_labels is None or roi_labels is None or cluster_order is None:
        plot.setTitle("Cluster-Colored Raster (No cluster data)")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI")
        return

    # Build label->cluster mapping
    label_to_cluster = {
        lbl: cls for lbl, cls in zip(roi_labels, cluster_labels)
    }

    # Query peak data from DB
    with Session(engine) as session:
        stmt = (
            select(ROI, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        roi_data = session.exec(stmt).all()

    if not roi_data:
        plot.setTitle("Cluster-Colored Raster (No ROI data)")
        return

    # Sort ROIs by cluster assignment
    roi_cluster_list = []
    for roi, data_analysis in roi_data:
        if roi.label_value not in label_to_cluster:
            continue
        if not data_analysis.peaks_den_dff:
            continue
        roi_cluster_list.append(
            (roi, data_analysis, label_to_cluster[roi.label_value])
        )

    roi_cluster_list.sort(key=lambda x: x[2])

    # Plot events per ROI
    y_labels = []
    for row_idx, (roi, data_analysis, cluster_id) in enumerate(roi_cluster_list):
        peaks = np.array(data_analysis.peaks_den_dff, dtype=float)
        if len(peaks) == 0:
            continue

        color = _get_cluster_color(cluster_id)
        scatter = pg.ScatterPlotItem(
            x=peaks,
            y=np.full(len(peaks), row_idx),
            pen=pg.mkPen(None),
            brush=pg.mkBrush(*color),
            symbol="s",
            size=3,
        )
        plot.addItem(scatter)
        y_labels.append(f"ROI {roi.label_value}")

    # Add legend entries (one per cluster)
    if n_clusters:
        legend = plot.addLegend(offset=(10, 10))
        for c in range(n_clusters):
            color = _get_cluster_color(c)
            legend.addItem(
                pg.ScatterPlotItem(
                    pen=pg.mkPen(None), brush=pg.mkBrush(*color),
                    symbol="s", size=8,
                ),
                f"Cluster {c}",
            )

    method_str = method or "unknown"
    plot.setTitle(f"Cluster-Colored Raster ({method_str}, k={n_clusters})")
    plot.setLabel("bottom", "Frames")
    plot.setLabel("left", "ROI (sorted by cluster)")
    plot.getAxis("left").setTicks([])


# ---- Plot 3: Cluster-Colored Traces ---- #

def _plot_cluster_colored_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Plot denoised ΔF/F traces colored by cluster assignment.

    Traces are normalized and vertically offset, sorted by cluster.
    Each trace is colored according to its cluster.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.invertY(False)
    vb.setAspectLocked(False)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get cluster data
    (
        _, roi_labels, cluster_labels, cluster_order,
        method, n_clusters, sil_score,
    ) = _get_cluster_data_from_db(engine, fov_name, run_id)

    if cluster_labels is None or roi_labels is None:
        plot.setTitle("Cluster-Colored Traces (No cluster data)")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "")
        return

    label_to_cluster = {
        lbl: cls for lbl, cls in zip(roi_labels, cluster_labels)
    }

    # Query traces from DB
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id)
                & (Traces.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        roi_data = session.exec(stmt).all()

    if not roi_data:
        plot.setTitle("Cluster-Colored Traces (No data)")
        return

    # Collect and sort by cluster
    trace_items = []
    for roi, traces in roi_data:
        if roi.label_value not in label_to_cluster:
            continue
        if traces.den_dff is None:
            continue
        trace_items.append(
            (roi, traces, label_to_cluster[roi.label_value])
        )

    trace_items.sort(key=lambda x: x[2])

    # Plot traces with vertical offset, colored by cluster
    offset = 0.0
    for roi, traces, cluster_id in trace_items:
        trace = np.asarray(traces.den_dff, dtype=float)
        if trace.size == 0:
            continue

        # Normalize: min-max scale to [0, 1]
        t_min, t_max = trace.min(), trace.max()
        if t_max > t_min:
            trace_norm = (trace - t_min) / (t_max - t_min)
        else:
            trace_norm = np.zeros_like(trace)

        color = _get_cluster_color(cluster_id)
        x = np.arrange(len(trace_norm))
        plot.plot(x, trace_norm + offset, pen=pg.mkPen(color, width=1))
        offset += 1.1  # vertical spacing between traces

    # Add legend
    if n_clusters:
        legend = plot.addLegend(offset=(10, 10))
        for c in range(n_clusters):
            color = _get_cluster_color(c)
            legend.addItem(
                pg.PlotDataItem(pen=pg.mkPen(color, width=2)),
                f"Cluster {c}",
            )

    method_str = method or "unknown"
    plot.setTitle(f"Cluster-Colored Traces ({method_str}, k={n_clusters})")
    plot.setLabel("bottom", "Frames")
    plot.setLabel("left", "Denoised ΔF/F (normalized, offset)")
    plot.getAxis("left").setTicks([])
```

---

## Step 9: Register plots in `_main_plot.py`

**File:** `src/cali/plot/_main_plot.py`

### 9a. Add import at top (with existing single-well imports):

```python
from ._single_wells_plots.cluster._plot_cluster_analysis import (
    _plot_cluster_colored_raster,
    _plot_cluster_colored_traces,
    _plot_cluster_sorted_correlation_heatmap,
)
```

### 9b. Register 3 new products (after the last existing AnalysisProduct, before any multi-well section):

```python
# Cluster Analysis Group
AnalysisProduct(
    name="Cluster-Sorted Correlation Heatmap",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cluster_sorted_correlation_heatmap,
    category="Cluster Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Cluster-Colored Calcium Peaks Raster",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cluster_colored_raster,
    category="Cluster Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Cluster-Colored Denoised ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cluster_colored_traces,
    category="Cluster Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
```

---

## Testing

### Unit test for `_cluster_analysis.py`

Create `tests/test_cluster_analysis.py`:

```python
"""Tests for cluster analysis module."""

import numpy as np
import pytest

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
    def test_returns_none_for_too_few_rois(self):
        corr = np.eye(2)
        result = compute_cluster_analysis(corr)
        assert result is None

    def test_hierarchical_fixed_k(self):
        corr = _make_block_corr_matrix(5, 3)  # 15 ROIs, 3 clusters
        result = compute_cluster_analysis(
            corr, method="hierarchical", n_clusters=3
        )
        assert result is not None
        assert result.n_clusters == 3
        assert len(result.labels) == 15
        assert set(result.labels) == {0, 1, 2}
        assert -1 <= result.silhouette_score <= 1
        assert len(result.order) == 15

    def test_kmeans_fixed_k(self):
        corr = _make_block_corr_matrix(5, 3)
        result = compute_cluster_analysis(
            corr, method="kmeans", n_clusters=3
        )
        assert result is not None
        assert result.n_clusters == 3
        assert len(result.labels) == 15

    def test_auto_detect_k(self):
        corr = _make_block_corr_matrix(5, 3)
        result = compute_cluster_analysis(
            corr, method="hierarchical", n_clusters=0, max_k=10
        )
        assert result is not None
        assert 2 <= result.n_clusters <= 10
        assert result.silhouette_score > 0  # block structure should cluster well

    def test_order_groups_clusters(self):
        corr = _make_block_corr_matrix(5, 3)
        result = compute_cluster_analysis(
            corr, method="hierarchical", n_clusters=3
        )
        assert result is not None
        # Verify that order sorts by cluster
        ordered_labels = [result.labels[i] for i in result.order]
        # Should be non-decreasing
        assert ordered_labels == sorted(ordered_labels)

    def test_silhouette_in_range(self):
        corr = _make_block_corr_matrix(10, 2)
        result = compute_cluster_analysis(
            corr, method="hierarchical", n_clusters=2
        )
        assert result is not None
        assert -1.0 <= result.silhouette_score <= 1.0

    def test_invalid_method_raises(self):
        corr = _make_block_corr_matrix(5, 2)
        with pytest.raises(ValueError, match="Unknown clustering method"):
            compute_cluster_analysis(corr, method="invalid")
```

---

## Summary of Algorithms

### Hierarchical Clustering (Ward's Linkage)
1. Compute distance: `d(i,j) = 1 - r(i,j)` where r is Pearson correlation
2. Convert to condensed form: `scipy.spatial.distance.squareform(dist)`
3. Linkage: `scipy.cluster.hierarchy.linkage(condensed, method='ward')`
4. Cut dendrogram: `scipy.cluster.hierarchy.fcluster(Z, t=k, criterion='maxclust')`

**Why Ward's**: Minimizes within-cluster variance, equivalent to K-means objective but produces a hierarchy. Gold standard in calcium imaging literature.

### K-means
1. Features: Each ROI's row in the correlation matrix (its correlation profile with all other ROIs)
2. `sklearn.cluster.KMeans(n_clusters=k, n_init=10, random_state=42)`
3. Clusters ROIs with similar correlation profiles together

**Why correlation features**: Using the NxN row as features captures each ROI's functional connectivity pattern.

### Silhouette Score (Auto-k)
1. For each k from 2 to max_k: cluster, compute `sklearn.metrics.silhouette_score(dist, labels, metric='precomputed')`
2. Pick k with highest silhouette score
3. Score range: [-1, 1]. >0.5 = good, >0.7 = strong

---

## Notes

- Clustering runs as part of FOV analysis (no separate pipeline step)
- Minimum 3 ROIs required (silhouette needs >= 2 clusters with >= 1 member each, and we need >= 2 clusters)
- The `cluster_order` field enables efficient sorted-heatmap rendering without re-computing at plot time
- All new DB fields are nullable (`| None`) so old databases work without migration
- scikit-learn is a well-maintained dependency already transitively present via scikit-image
