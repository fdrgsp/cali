"""Dimensionality reduction utilities for multi-well per-condition analysis.

This module builds a per-FOV feature matrix from scalar calcium imaging metrics
and applies PCA for visualisation.

The feature matrix has one row per FOV and includes:

- Per-ROI scalars averaged to FOV level (amplitude, frequency, IEI, spike freq,
  cell size, % active).
- FOVAnalysis burst stats (burst count, avg duration, avg interval).

Usage
-----
>>> from sqlalchemy import create_engine
>>> engine = create_engine("sqlite:///my.cali")
>>> df = build_fov_feature_matrix(engine, run_id=1)
>>> coords, pca = compute_pca(df)
>>> # coords: np.ndarray shape (n_fovs, 2); color by df["condition"]

Notes
-----
- Columns with all-NaN values are dropped before fitting.
- Remaining NaN values are imputed with the column median.
- All features are z-scored (StandardScaler) before PCA.

References
----------
- Stringer et al. (2019, Science) — PCA for neural population data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

# Feature columns in the matrix (order matters for PCA loadings display)
FEATURE_COLUMNS = [
    "mean_amplitude",
    "mean_frequency",
    "mean_iei",
    "mean_spike_freq",
    "mean_spike_freq_edges",
    "mean_cell_size",
    "pct_active",
    "burst_count",
    "burst_avg_duration_s",
    "burst_avg_interval_s",
]


def build_fov_feature_matrix(
    engine: Engine,
    run_id: int | None = None,
) -> "import pandas as pd; pd.DataFrame":  # type: ignore[return]
    """Build a per-FOV feature matrix from the database.

    One row per FOV.  Columns are :data:`FEATURE_COLUMNS` plus ``"fov_name"``
    and ``"condition"`` (used for colouring downstream visualisations).

    Parameters
    ----------
    engine : Engine
        Database engine.
    run_id : int | None
        CaliResult id.  When ``None`` all results in the DB are included.

    Returns
    -------
    pandas.DataFrame
        Shape ``(n_fovs, len(FEATURE_COLUMNS) + 2)`` with columns
        ``["fov_name", "condition"] + FEATURE_COLUMNS``.
    """
    import pandas as pd
    from sqlmodel import Session, col, select

    from cali.sqlmodel import (
        FOV,
        ROI,
        FOVAnalysis,
        DataAnalysis,
        Well,
    )

    from ._util import _get_condition_label

    rows: list[dict] = []

    with Session(engine) as session:
        # ------------------------------------------------------------------
        # 1.  Per-FOV means of ROI-level scalars
        # ------------------------------------------------------------------
        roi_stmt = (
            select(DataAnalysis, ROI, FOV, Well)
            .join(ROI, DataAnalysis.roi_id == ROI.id)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if run_id is not None:
            roi_stmt = roi_stmt.where(
                col(DataAnalysis.analysis_result_id) == run_id
            )

        roi_results = session.exec(roi_stmt).all()

        # Aggregate per-ROI scalars → per-FOV accumulators
        fov_roi_data: dict[int, dict[str, list[float]]] = {}
        fov_meta: dict[int, tuple[str, str]] = {}  # fov_id → (fov_name, condition)

        for analysis, roi, fov, well in roi_results:
            if fov.id not in fov_roi_data:
                fov_roi_data[fov.id] = {k: [] for k in FEATURE_COLUMNS}
                fov_meta[fov.id] = (fov.name, _get_condition_label(well))

            d = fov_roi_data[fov.id]

            # amplitude: mean per ROI
            if analysis.peaks_amplitudes_den_dff:
                d["mean_amplitude"].append(
                    float(np.mean(analysis.peaks_amplitudes_den_dff))
                )

            # frequency
            if analysis.den_dff_frequency is not None:
                d["mean_frequency"].append(float(analysis.den_dff_frequency))

            # IEI: mean per ROI
            if analysis.iei:
                d["mean_iei"].append(float(np.mean(analysis.iei)))

            # spike frequency (thresholded)
            if analysis.inferred_spikes_frequency is not None:
                d["mean_spike_freq"].append(
                    float(analysis.inferred_spikes_frequency)
                )

            # spike frequency (rising edges)
            if analysis.inferred_spikes_rising_edge_frequency is not None:
                d["mean_spike_freq_edges"].append(
                    float(analysis.inferred_spikes_rising_edge_frequency)
                )

            # cell size
            if roi.cell_size is not None:
                d["mean_cell_size"].append(float(roi.cell_size))

        # Count active/total ROIs per FOV (for pct_active)
        all_roi_stmt = (
            select(ROI, FOV)
            .join(FOV, ROI.fov_id == FOV.id)
        )
        if run_id is not None:
            all_roi_stmt = all_roi_stmt.join(
                DataAnalysis, DataAnalysis.roi_id == ROI.id
            ).where(col(DataAnalysis.analysis_result_id) == run_id)

        fov_total: dict[int, int] = {}
        fov_active: dict[int, int] = {}
        for roi, fov in session.exec(all_roi_stmt).all():
            fov_total[fov.id] = fov_total.get(fov.id, 0) + 1
            if roi.active:
                fov_active[fov.id] = fov_active.get(fov.id, 0) + 1

        # ------------------------------------------------------------------
        # 2.  FOVAnalysis burst stats (one scalar per FOV)
        # ------------------------------------------------------------------
        fov_stmt = (
            select(FOVAnalysis, FOV, Well)
            .join(FOV, FOVAnalysis.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
        )
        if run_id is not None:
            fov_stmt = fov_stmt.where(
                col(FOVAnalysis.analysis_result_id) == run_id
            )

        fov_analysis_map: dict[int, FOVAnalysis] = {}
        for fa, fov, well in session.exec(fov_stmt).all():
            fov_analysis_map[fov.id] = fa
            if fov.id not in fov_meta:
                fov_meta[fov.id] = (fov.name, _get_condition_label(well))

        # ------------------------------------------------------------------
        # 3.  Assemble row per FOV
        # ------------------------------------------------------------------
        all_fov_ids = set(fov_roi_data.keys()) | set(fov_analysis_map.keys())

        for fov_id in sorted(all_fov_ids):
            fov_name, condition = fov_meta.get(fov_id, (str(fov_id), "unknown"))
            row: dict = {"fov_name": fov_name, "condition": condition}

            # ROI-level scalars → FOV mean
            d = fov_roi_data.get(fov_id, {})
            for col_name in [
                "mean_amplitude",
                "mean_frequency",
                "mean_iei",
                "mean_spike_freq",
                "mean_spike_freq_edges",
                "mean_cell_size",
            ]:
                vals = d.get(col_name, [])
                row[col_name] = float(np.mean(vals)) if vals else float("nan")

            # % active
            total = fov_total.get(fov_id, 0)
            active = fov_active.get(fov_id, 0)
            row["pct_active"] = (active / total * 100.0) if total > 0 else float("nan")

            # Burst stats from FOVAnalysis
            fa = fov_analysis_map.get(fov_id)
            if fa is not None:
                row["burst_count"] = (
                    float(fa.spike_burst_count)
                    if fa.spike_burst_count is not None
                    else float("nan")
                )
                row["burst_avg_duration_s"] = (
                    float(fa.spike_burst_avg_duration)
                    if fa.spike_burst_avg_duration is not None
                    else float("nan")
                )
                row["burst_avg_interval_s"] = (
                    float(fa.spike_burst_avg_interval)
                    if fa.spike_burst_avg_interval is not None
                    else float("nan")
                )
            else:
                row["burst_count"] = float("nan")
                row["burst_avg_duration_s"] = float("nan")
                row["burst_avg_interval_s"] = float("nan")

            rows.append(row)

    return pd.DataFrame(rows)


def _prepare_feature_matrix(
    df: "import pandas as pd; pd.DataFrame",  # type: ignore[type-arg]
    feature_cols: list[str] | None = None,
) -> "import numpy as np; np.ndarray":
    """Impute NaN (column median) and z-score feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix from :func:`build_fov_feature_matrix`.
    feature_cols : list[str] | None
        Columns to use.  Defaults to :data:`FEATURE_COLUMNS` minus any
        all-NaN columns.

    Returns
    -------
    np.ndarray
        Scaled feature matrix of shape ``(n_fovs, n_features)``.
    list[str]
        Column names of the final feature set (all-NaN columns dropped).
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Drop all-NaN columns
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    X = df[feature_cols].copy()
    # Impute with column median
    for col in feature_cols:
        median = X[col].median()
        X[col] = X[col].fillna(median)

    return StandardScaler().fit_transform(X.values), feature_cols


def compute_pca(
    df: "import pandas as pd; pd.DataFrame",  # type: ignore[type-arg]
    feature_cols: list[str] | None = None,
    n_components: int = 2,
) -> "tuple[np.ndarray, object, list[str]]":
    """Run PCA on the FOV feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix from :func:`build_fov_feature_matrix`.
    feature_cols : list[str] | None
        Subset of feature columns to use (default: all non-NaN columns).
    n_components : int
        Number of PCA components (default 2 for 2-D scatter plot).

    Returns
    -------
    coords : np.ndarray
        Shape ``(n_fovs, n_components)``.  Row order matches ``df``.
    pca : sklearn.decomposition.PCA
        Fitted PCA object.  Access ``pca.explained_variance_ratio_`` and
        ``pca.components_`` for scree / loading plots.
    used_features : list[str]
        Feature columns that were actually used (all-NaN columns excluded).
    """
    from sklearn.decomposition import PCA

    X, used_features = _prepare_feature_matrix(df, feature_cols)
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    coords: np.ndarray = pca.fit_transform(X)
    return coords, pca, used_features


# ---------------------------------------------------------------------------
# Scatter plot helpers (pyqtgraph output)
# ---------------------------------------------------------------------------


def _render_scatter(
    widget: "object",  # _MultilWellGraphWidget
    coords: "np.ndarray",
    conditions: "list[str]",
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Render a 2-D scatter plot coloured by condition into *widget*."""
    import pyqtgraph as pg

    from ._util import _get_default_conditions

    unique_conditions = list(dict.fromkeys(conditions))  # preserve order, deduplicate

    cond_list: dict[str, dict[str, bool | str]] = widget.conditions  # type: ignore[attr-defined]
    if not cond_list or set(cond_list.keys()) != set(unique_conditions):
        cond_list = _get_default_conditions(unique_conditions)
        widget.conditions = cond_list  # type: ignore[attr-defined]

    plot_item = widget.plot_item  # type: ignore[attr-defined]

    # Legend — create once, clear stale entries
    legend = plot_item.addLegend(offset=(-10, 5))
    legend.clear()

    cond_arr = np.asarray(conditions)
    for cond, cond_opts in cond_list.items():
        if not cond_opts.get("visible", True):
            continue
        mask = cond_arr == cond
        x = coords[mask, 0]
        y = coords[mask, 1]
        color = cond_opts.get("color", "gray")
        scatter = pg.ScatterPlotItem(
            x=x,
            y=y,
            size=12,
            pen=pg.mkPen("k", width=0.5),
            brush=pg.mkBrush(color),
            symbol="o",
        )
        plot_item.addItem(scatter)
        legend.addItem(scatter, cond)

    plot_item.setLabel("bottom", x_label)
    plot_item.setLabel("left", y_label)
    plot_item.setTitle(title)
    plot_item.showGrid(x=True, y=True, alpha=0.3)


def plot_pca_scatter(
    widget: "object",  # _MultilWellGraphWidget
    text: str,
    engine: "Engine",
    run_id: int | None = None,
) -> None:
    """Plot a PCA scatter of FOVs coloured by condition.

    Builds the per-FOV feature matrix from *engine*, z-scores all features,
    fits a 2-component PCA, and renders one scatter point per FOV coloured by
    its condition.

    Axis labels include the explained-variance percentage for PC1 and PC2.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        Target plot widget.
    text : str
        Plot name (used in the title).
    engine : Engine
        Database engine.
    run_id : int | None
        Filter to a single CaliResult; ``None`` uses all runs in the DB.
    """
    try:
        df = build_fov_feature_matrix(engine, run_id)
    except Exception:
        widget.clear_plot()  # type: ignore[attr-defined]
        return

    if df is None or len(df) < 2:
        widget.clear_plot()  # type: ignore[attr-defined]
        return

    try:
        coords, pca, _ = compute_pca(df)
    except Exception:
        widget.clear_plot()  # type: ignore[attr-defined]
        return

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100 if len(pca.explained_variance_ratio_) > 1 else 0.0

    _render_scatter(
        widget=widget,
        coords=coords,
        conditions=df["condition"].tolist(),
        x_label=f"PC1 ({var1:.1f}% var)",
        y_label=f"PC2 ({var2:.1f}% var)",
        title="PCA — FOV Feature Space",
    )
