# Multi-Well Per-Condition Aggregation Plan

## Overview

The goal is to aggregate all per-FOV and per-ROI calcium imaging metrics across wells that share the
same condition label (from `Plate.plate_maps` / `WellCondition`). The aggregation hierarchy is:

```
ROI → FOV → Well → Condition
```

**The unit of statistical replication is the FOV** (not the individual ROI), to avoid
pseudoreplication — a canonical requirement in cellular calcium imaging analysis. Each FOV is an
independent field of view; ROIs within a FOV are not independent observations. Where a well has
multiple FOVs, the FOV is still the unit, and the well is an intermediate grouping. All error bars
represent pooled SEM across FOVs per condition unless stated otherwise.

---

## Data Retrieval Strategy (shared across all metrics)

All queries follow the same pattern:

```python
# Pseudocode — adapt with SQLModel session
{condition_label: {fov_id: [scalars_or_matrix]}}
```

The grouping key is always obtained via `_get_condition_label(well)` from
`_multi_wells_plots/_util.py`, which joins all `Condition.name` values attached to the well. This
already handles multi-dimensional plate maps (e.g. genotype × treatment).

---

## 1  Scalar per-ROI Metrics (Amplitude, Frequency, IEI, Spike Frequency, Cell Size)

**Applies to:** `peaks_amplitudes_den_dff` (list/ROI, flatten to mean), `den_dff_frequency`
(scalar/ROI), `iei` (list/ROI, flatten to mean), `inferred_spikes_frequency` (scalar/ROI),
`inferred_spikes_rising_edge_frequency` (scalar/ROI), `roi.cell_size` (scalar/ROI).

### Aggregation Protocol

**Step 1 — ROI → FOV**

For each FOV, collect all *active* ROI values (filter `roi.active == True`):

```python
import numpy as np

fov_mean = np.mean(roi_values)          # central tendency
fov_sem  = np.std(roi_values, ddof=1) / np.sqrt(len(roi_values))
fov_n    = len(roi_values)
```

For `peaks_amplitudes_den_dff` and `iei` (which are per-ROI lists), first reduce each ROI to its
own mean, then treat that as the per-ROI scalar:

```python
roi_scalar = np.mean(roi_list)   # mean amplitude or mean IEI per ROI
```

**Step 2 — FOV → Condition (weighted mean + pooled SEM)**

Collect `(fov_mean_i, fov_sem_i, fov_n_i)` for every FOV that belongs to the condition:

```python
total_n     = sum(fov_ns)
w_mean      = np.average(fov_means, weights=fov_ns)          # weighted mean
pooled_var  = sum(n_i * (sem_i**2 + (m_i - w_mean)**2)       # Welch-like pooled variance
              for m_i, sem_i, n_i in zip(fov_means, fov_sems, fov_ns)) / total_n
pooled_sem  = np.sqrt(pooled_var / total_n)
```

This is the standard *pooled SEM* used in electrophysiology and calcium imaging meta-analysis
(Bhattacharya et al., 2022; Stringer et al., 2019 supplementary). The formula propagates both
within-FOV variability (captured by `fov_sem`) and between-FOV variability.

**Scalar summary output per condition:**

```python
from dataclasses import dataclass

@dataclass
class ConditionStats:
    condition: str
    weighted_mean: float
    pooled_sem: float
    total_n_rois: int
    n_fovs: int
    distribution: list[float]   # all per-FOV means, for violin/swarm overlay
```

**Why FOV and not well?** If a well has k FOVs, averaging the well mean before the condition mean
would down-weight FOVs from wells with fewer FOVs. Weighting by `n_rois` per FOV is the most
information-preserving approach and is consistent with what `scipy.stats.sem` computes when
treating ROIs pooled within a condition.

---

## 2  Pearson Correlation Matrices (ΔF/F and denoised ΔF/F)

**Fields:** `FOVAnalysis.calcium_dff_correlation_matrix`,
`FOVAnalysis.calcium_den_dff_corr_matrix`.

### Key Principle

Pearson *r* values are **not** linearly additive — their sampling distribution is skewed, especially
near ±1. The canonical solution (canonical since Fisher 1921, standard in neuroscience) is the
**Fisher z-transformation** before averaging:

```python
z = np.arctanh(r)    # z ~ N(0, 1/(N-3)) for large N
r_mean = np.tanh(np.mean(z_values))   # back-transform
```

### Aggregation Protocol

**Step 1 — Within-FOV summary**

For each FOV:
- Take the NxN correlation matrix.
- Extract upper-triangle (excluding diagonal): all `n*(n-1)/2` pairwise values.
- Fisher z-transform all values.
- `fov_z_mean = np.mean(z_values)`
- `fov_z_sem  = np.std(z_values, ddof=1) / np.sqrt(len(z_values))`
- `fov_n_pairs = len(z_values)`
- Back-transform for reporting: `fov_r_mean = np.tanh(fov_z_mean)`.

**Step 2 — FOV → Condition**

Weighted mean ± pooled SEM in z-space (same formula as §1, using FOV z-means and z-SEMs):

```python
cond_z_mean  = np.average(fov_z_means, weights=fov_n_pairs)
# pooled SEM in z-space using Welch-like formula
cond_r_mean  = np.tanh(cond_z_mean)   # report in r-space
```

### Functional Connectivity Summary per Condition

A richer network-level summary (commonly reported in calcium imaging connectivity papers, e.g.
Avitan et al., 2017; Bhattacharya et al., 2022):

For each FOV, threshold the correlation matrix to build a binary adjacency matrix:

```python
# Threshold options (expose as parameter):
# 1. Fixed threshold, e.g. r > 0.3
# 2. Significance threshold: keep pairs where p < 0.05 (scipy.stats.pearsonr per pair)
# 3. Top-k% of pairs

adjacency = corr_matrix > threshold   # NxN bool, zero diagonal
```

Then compute per-FOV graph metrics using `networkx`:

```python
import networkx as nx

G = nx.from_numpy_array(adjacency.astype(float))
graph_metrics = {
    "mean_degree":              np.mean([d for _, d in G.degree()]),
    "density":                  nx.density(G),
    "global_clustering_coeff":  nx.transitivity(G),    # fraction of closed triplets
    "mean_clustering_coeff":    nx.average_clustering(G),
    "n_connected_components":   nx.number_connected_components(G),
}
```

Each of these is then a scalar per FOV → aggregate to condition with §1 protocol.

**Output per condition:**

```python
@dataclass
class ConditionConnectivity:
    condition: str
    mean_pairwise_r: float          # back-transformed Fisher z mean
    pooled_r_sem: float
    mean_degree: float
    mean_degree_sem: float
    density: float
    density_sem: float
    global_clustering: float
    global_clustering_sem: float
    n_fovs: int
```

---

## 3  Clustering (k-means / Hierarchical)

**Fields:** `FOVAnalysis.cluster_labels`, `cluster_silhouette_score`, `cluster_n_clusters`.

Clustering is inherently per-FOV and cell identity is **not** preserved across FOVs (ROI label 5 in
FOV-A is not the same cell as ROI label 5 in FOV-B). Two scientifically valid strategies:

### Strategy A — Aggregate Summary Scalars per Condition (default)

Treat each FOV's clustering result as producing summary scalars:

```python
fov_cluster_stats = {
    "silhouette_score":          fov_analysis.cluster_silhouette_score,
    "n_clusters":                fov_analysis.cluster_n_clusters,
    "largest_cluster_fraction":  max(Counter(labels).values()) / len(labels),
}
```

Aggregate silhouette score and cluster size fractions per condition with §1 protocol (weighted mean
± pooled SEM). This is the approach used in Bhattacharya et al. 2022 to compare functional
modular structure across conditions.

### Strategy B — Co-occurrence Matrix (opt-in deeper metric)

A richer approach that preserves network structure without requiring matched cell identities:

1. For each FOV, compute the **co-occurrence matrix**: `C[i,j] = 1` if ROIs i and j are in the
   same cluster, else 0. This is a symmetric NxN binary matrix.
2. Compute **intra-cluster ratio**: fraction of all pairs that cluster together.
3. Compare this intra-cluster ratio across conditions (mean ± SEM).

This approach is adapted from consensus clustering (Monti et al., 2003) as a summary of functional
modularity.

```python
from itertools import combinations

def co_occurrence_intra_ratio(labels: list[int]) -> float:
    same  = sum(1 for a, b in combinations(labels, 2) if a == b)
    total = len(labels) * (len(labels) - 1) // 2
    return same / total if total > 0 else 0.0
```

---

## 4  CCG Analysis (Inferred Spikes Cross-Correlogram)

**Fields (per FOV):**
- `spike_max_lag_correlation_matrix` / `_rising_edges` — NxN CCG peak correlation
- `spike_max_lag_values_matrix` / `_rising_edges` — NxN lag frames at CCG peak
- `spike_ccg_zscore_matrix` / `_rising_edges` — NxN z-score for significance
- `global_spike_max_lag_correlation` / `_rising_edges` — median off-diagonal (already scalar)
- `spike_jitter_synchrony_matrix` / `_rising_edges` — NxN jitter synchrony
- `global_spike_jitter_synchrony` / `_rising_edges` — scalar summary

### Aggregation Protocol

**Already-scalar globals** (`global_spike_max_lag_correlation`, `global_spike_jitter_synchrony`):
aggregate directly with the §1 FOV → condition protocol. Since these are correlation coefficients,
apply Fisher z before averaging:

```python
z_global = np.arctanh(np.clip(global_corr, -0.9999, 0.9999))
# average in z-space across FOVs, back-transform for reporting
```

**Full NxN matrices** — same as §2:
- Extract upper-triangle pairs.
- Fisher z-transform `spike_max_lag_correlation` values.
- `fov_mean_ccg_r = np.tanh(np.mean(z_pairs))`.

**Significant connectivity fraction** (from z-score matrix):

```python
# A pair is "significantly synchronized" if |z-score| > 2.58 (p < 0.01, two-tailed)
sig_fraction = np.mean(np.abs(np.triu(zscore_matrix, k=1)) > 2.58)
```

One scalar per FOV → aggregate with §1 protocol.

**Max-lag values** (frames, not correlations):
- No Fisher z needed; mean ± SEM of upper-triangle values.
- Report as milliseconds: `lag_ms = lag_frames / frame_rate * 1000`.

**Output per condition:**

```python
@dataclass
class ConditionCCG:
    condition: str
    mean_ccg_r: float                     # back-transformed Fisher z mean
    ccg_r_sem: float
    mean_jitter_synchrony: float
    jitter_synchrony_sem: float
    sig_pair_fraction: float              # fraction of significantly correlated pairs
    sig_pair_fraction_sem: float
    mean_lag_ms: float                    # mean lag at CCG peak
    mean_lag_ms_sem: float
    n_fovs: int
    # same fields with _edges suffix for rising-edge analysis
```

---

## 5  PCA / UMAP Dimensionality Reduction

Complementary to all of the above: allows visual clustering of FOVs (or conditions) in a
low-dimensional space. Increasingly standard in calcium imaging meta-analyses to reveal systematic
differences across conditions (Stringer et al., 2019; Bhattacharya et al., 2022).

### Feature Matrix Construction

Two levels of analysis:

#### 5a  FOV-level (recommended for condition comparison)

One row per FOV. Features (z-score each column before PCA/UMAP):

| Feature | Source |
|---|---|
| `mean_amplitude` | mean of per-ROI amplitudes within FOV |
| `mean_frequency` | mean of per-ROI `den_dff_frequency` |
| `mean_iei` | mean of per-ROI mean-IEI |
| `mean_spike_freq` | mean of per-ROI `inferred_spikes_frequency` |
| `mean_spike_freq_edges` | mean of per-ROI `inferred_spikes_rising_edge_frequency` |
| `mean_cell_size` | mean of per-ROI `cell_size` |
| `pct_active` | fraction of active ROIs |
| `mean_pairwise_r` | Fisher-z-averaged Pearson r (§2) |
| `sig_pair_fraction` | fraction of significantly correlated pairs (§4) |
| `mean_jitter_synchrony` | `global_spike_jitter_synchrony` |
| `mean_ccg_r` | Fisher-z-averaged CCG r (§4) |
| `silhouette_score` | `cluster_silhouette_score` (§3) |
| `burst_count` | `spike_burst_count` |
| `burst_avg_duration` | `spike_burst_avg_duration` |
| `burst_avg_interval` | `spike_burst_avg_interval` |

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

feature_df = pd.DataFrame(rows)   # rows = one dict per FOV
X = StandardScaler().fit_transform(feature_df[feature_cols])
pca = PCA(n_components=2)
coords = pca.fit_transform(X)     # shape (n_fovs, 2), color by condition
```

#### 5b  ROI-level (for single-cell exploration)

One row per *active* ROI. Features: amplitude, frequency, mean IEI, spike frequency, cell size.
Color by condition (propagated well → FOV → ROI). Useful for checking whether individual cells
differ across conditions independent of network structure.

### UMAP

UMAP is preferred over t-SNE for calcium imaging data because it preserves global structure
(Becht et al., 2019) and is increasingly the standard for single-cell and neural data exploration:

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
coords = reducer.fit_transform(X)
```

Use FOV-level feature matrix to compare **conditions** (not individual cells); use ROI-level for
cell-level heterogeneity. Both retain condition color as metadata.

Always report for PCA:
- Scree plot: `pca.explained_variance_ratio_`
- PC1/PC2 loadings: `pca.components_` to identify which features drive separation

For UMAP, report `n_neighbors` and `min_dist` as hyperparameters (no formal explained variance).

---

## Implementation Modules

Proposed new / modified files — all in `src/cali/plot/_multi_wells_plots/`:

| File | Responsibility |
|---|---|
| `_aggregation.py` (new) | All aggregation functions: §1 scalar protocol, §2 Fisher-z correlation, §3 cluster stats, §4 CCG stats, §5 feature matrix builder. Pure functions; no DB queries. |
| `_db_queries.py` (new) | All per-condition DB query functions (extract raw data from DB → dicts grouped by condition/FOV). Separates DB concerns from math. |
| `_calcium_peaks.py` (update) | Uncomment and refactor amplitude/frequency/IEI bar plots using `_aggregation.py`. |
| `_spike_analysis.py` (update) | Uncomment synchrony/burst bar plots; add CCG aggregation output. |
| `_cell_properties.py` (update) | Cell size already partially working; verify uses `ConditionStats` from `_aggregation.py`. |
| `_connectivity.py` (new) | Per-condition functional connectivity (§2 graph metrics + §4 CCG). |
| `_dimensionality_reduction.py` (new) | PCA/UMAP feature matrix builder and sklearn/umap calls (§5). |
| `_main_plot.py` (update) | Uncomment and register all new `AnalysisProduct` entries under `AnalysisGroup.MULTI_WELL`. |

---

## Aggregation Decisions Summary

| Parameter | Aggregate per | Error metric | Transform |
|---|---|---|---|
| Amplitude, Frequency, IEI, Spike freq, Cell size | FOV (weighted mean of ROIs) | Pooled SEM across FOVs | None |
| Pearson r (dff, den_dff) | FOV upper-triangle → condition | Pooled SEM in z-space | Fisher arctanh |
| CCG max-lag correlation | FOV upper-triangle → condition | Pooled SEM in z-space | Fisher arctanh |
| CCG z-score (sig pair fraction) | FOV scalar → condition | SEM across FOVs | None |
| Max-lag values (frames / ms) | FOV upper-triangle → condition | Pooled SEM | None (linear) |
| Jitter synchrony (global) | FOV scalar → condition | SEM across FOVs | Fisher arctanh |
| Cluster silhouette score | FOV scalar → condition | SEM across FOVs | None |
| Cluster co-occurrence ratio | FOV scalar → condition | SEM across FOVs | None |
| PCA / UMAP | FOV feature vector | — (visualization) | StandardScaler z-score |

---

## References

- Fisher (1921) — z-transformation for averaging Pearson r.
- Bhattacharya et al. (2022, *Cell Reports*) — calcium imaging network analysis, FOV as unit of replication.
- Stringer et al. (2019, *Science*) — dimensionality of neural population activity, PCA/UMAP usage.
- Becht et al. (2019, *Nature Biotechnology*) — UMAP vs t-SNE for biological data.
- Monti et al. (2003, *Machine Learning*) — consensus clustering / co-occurrence matrix.
- Avitan et al. (2017, *Neuron*) — functional connectivity thresholding in calcium imaging.
