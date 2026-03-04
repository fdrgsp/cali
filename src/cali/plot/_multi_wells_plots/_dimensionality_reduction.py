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

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
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
    include_stim_status: bool = False,
) -> pd.DataFrame:
    """Build a per-FOV feature matrix from the database.

    One row per FOV.  Columns are :data:`FEATURE_COLUMNS` plus `"fov_name"`
    and `"condition"` (used for colouring downstream visualisations).

    Parameters
    ----------
    engine : Engine
        Database engine.
    run_id : int | None
        CaliResult id.  When `None` all results in the DB are included.
    include_stim_status : bool
        When True, split each FOV into two rows — one for stimulated ROIs and
        one for non-stimulated ROIs — and append the stim/non-stim suffix to
        the condition label.  Only meaningful for Evoked Activity runs whose
        ROIs have a `stimulated` attribute set.

    Returns
    -------
    pandas.DataFrame
        Shape `(n_rows, len(FEATURE_COLUMNS) + 2)` with columns
        `["fov_name", "condition"] + FEATURE_COLUMNS`.
    """
    import pandas as pd
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, ROI, DataAnalysis, FOVAnalysis, Well

    from ._util import _get_condition_label, _get_experiment_type

    rows: list[dict] = []

    with Session(engine) as session:
        # ------------------------------------------------------------------
        # 0.  Optionally look up experiment_type for stim-split labelling
        # ------------------------------------------------------------------
        experiment_type: str | None = None
        if include_stim_status and run_id is not None:
            experiment_type = _get_experiment_type(session, run_id)

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
            roi_stmt = roi_stmt.where(col(DataAnalysis.analysis_result_id) == run_id)

        roi_results = session.exec(roi_stmt).all()

        # Aggregate per-ROI scalars → per-FOV (or per-FOV+stim) accumulators.
        # When include_stim_status=True the key is (fov_id, roi.stimulated)
        # so that stim and non-stim ROIs within the same FOV end up in
        # separate rows of the feature matrix.
        FovKey: TypeAlias = tuple  # (fov_id,) or (fov_id, stim_status)
        fov_roi_data: dict[FovKey, dict[str, list[float]]] = {}
        fov_meta: dict[FovKey, tuple[str, str]] = {}  # key → (fov_name, condition)

        def _make_key(fov_id: int, roi: ROI) -> FovKey:
            if include_stim_status:
                return (fov_id, roi.stimulated)
            return (fov_id,)

        for analysis, roi, fov, well in roi_results:
            key = _make_key(fov.id, roi)
            if key not in fov_roi_data:
                fov_roi_data[key] = {k: [] for k in FEATURE_COLUMNS}
                if include_stim_status:
                    label = _get_condition_label(well, roi, experiment_type)
                else:
                    label = _get_condition_label(well)
                stim_suffix = ""
                if include_stim_status and roi.stimulated is not None:
                    stim_suffix = "_stim" if roi.stimulated else "_non_stim"
                fov_meta[key] = (f"{fov.name}{stim_suffix}", label)

            d = fov_roi_data[key]

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
                d["mean_spike_freq"].append(float(analysis.inferred_spikes_frequency))

            # spike frequency (rising edges)
            if analysis.inferred_spikes_rising_edge_frequency is not None:
                d["mean_spike_freq_edges"].append(
                    float(analysis.inferred_spikes_rising_edge_frequency)
                )

            # cell size
            if roi.cell_size is not None:
                d["mean_cell_size"].append(float(roi.cell_size))

        # Count active/total ROIs per FOV (for pct_active).
        # When include_stim_status=True, count separately per (fov_id, stim).
        all_roi_stmt = select(ROI, FOV).join(FOV, ROI.fov_id == FOV.id)
        if run_id is not None:
            all_roi_stmt = all_roi_stmt.join(
                DataAnalysis, DataAnalysis.roi_id == ROI.id
            ).where(col(DataAnalysis.analysis_result_id) == run_id)

        fov_total: dict = {}
        fov_active: dict = {}
        for roi, fov in session.exec(all_roi_stmt).all():
            k = _make_key(fov.id, roi) if include_stim_status else fov.id
            fov_total[k] = fov_total.get(k, 0) + 1
            if roi.active:
                fov_active[k] = fov_active.get(k, 0) + 1

        # ------------------------------------------------------------------
        # 2.  FOVAnalysis burst stats (one scalar per FOV)
        # ------------------------------------------------------------------
        fov_stmt = (
            select(FOVAnalysis, FOV, Well)
            .join(FOV, FOVAnalysis.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
        )
        if run_id is not None:
            fov_stmt = fov_stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)

        fov_analysis_map: dict[int, FOVAnalysis] = {}
        for fa, fov, well in session.exec(fov_stmt).all():
            fov_analysis_map[fov.id] = fa
            # Register plain fov.id key for fov_meta when not stim-split
            if not include_stim_status:
                plain_key = (fov.id,)
                if plain_key not in fov_meta:
                    fov_meta[plain_key] = (fov.name, _get_condition_label(well))

        # ------------------------------------------------------------------
        # 3.  Assemble one row per key (fov or fov+stim)
        # ------------------------------------------------------------------
        all_keys = set(fov_roi_data.keys())
        # Also include FOVs that only have burst stats (no active ROIs)
        if not include_stim_status:
            for fov_id in fov_analysis_map:
                all_keys.add((fov_id,))

        for key in sorted(
            all_keys, key=lambda k: (k[0], str(k[1]) if len(k) > 1 else "")
        ):
            fov_id = key[0]
            fov_name, condition = fov_meta.get(key, (str(fov_id), "unknown"))
            row: dict = {"fov_name": fov_name, "condition": condition}

            # ROI-level scalars → mean
            d = fov_roi_data.get(key, {})
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

            # % active — use stim-aware key when splitting
            pct_key = key if include_stim_status else fov_id
            total = fov_total.get(pct_key, 0)
            active = fov_active.get(pct_key, 0)
            row["pct_active"] = (active / total * 100.0) if total > 0 else float("nan")

            # Burst stats from FOVAnalysis (FOV-level).
            # When stim-splitting, burst stats are shared across stim/non-stim
            # rows for the same FOV, which would create artificial correlation
            # in PCA.  Exclude them from stim-split matrices by setting to NaN.
            if include_stim_status:
                row["burst_count"] = float("nan")
                row["burst_avg_duration_s"] = float("nan")
                row["burst_avg_interval_s"] = float("nan")
            else:
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
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
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
        Scaled feature matrix of shape `(n_fovs, n_features)`.
    list[str]
        Column names of the final feature set (all-NaN columns dropped).
    """
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

    # Drop zero-variance columns (constant after imputation) to avoid
    # division-by-zero in StandardScaler and PCA explained_variance_ratio_.
    keep = [i for i, c in enumerate(feature_cols) if X[c].nunique() > 1]
    feature_cols = [feature_cols[i] for i in keep]
    X = X[feature_cols]

    if not feature_cols:
        raise ValueError("No features with non-zero variance remain after filtering.")

    return StandardScaler().fit_transform(X.values), feature_cols


def compute_pca(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_components: int = 2,
) -> tuple[np.ndarray, Any, list[str]]:
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
        Shape `(n_fovs, n_components)`.  Row order matches `df`.
    pca : sklearn.decomposition.PCA
        Fitted PCA object.  Access `pca.explained_variance_ratio_` and
        `pca.components_` for scree / loading plots.
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
    widget: object,  # _MultilWellGraphWidget
    coords: np.ndarray,
    conditions: list[str],
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
        # PCA scatter always uses multicolor so different conditions are
        # visually distinguishable in the scatter space.
        cond_list = _get_default_conditions(unique_conditions, multicolor=True)
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


def _run_pca_scatter(
    widget: object,
    engine: Engine,
    run_id: int | None,
    include_stim_status: bool,
    title: str,
) -> None:
    """Shared implementation for PCA scatter plots."""
    import logging

    logger = logging.getLogger(__name__)

    try:
        df = build_fov_feature_matrix(engine, run_id, include_stim_status)
    except Exception:
        logger.debug("Failed to build FOV feature matrix for PCA", exc_info=True)
        widget.clear_plot()  # type: ignore[attr-defined]
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            f"{title}<br><span style='color:red; font-size:9pt'>"
            "Error building feature matrix</span>"
        )
        return

    if df is None or len(df) < 2:
        widget.clear_plot()  # type: ignore[attr-defined]
        n = 0 if df is None else len(df)
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            f"{title}<br><span style='color:magenta; font-size:9pt'>"
            f"Need ≥ 2 FOVs for PCA (found {n})</span>"
        )
        return

    # Read user-selected PCA features from the widget (if any)
    pca_features: list[str] | None = getattr(widget, "_pca_features", None)

    try:
        coords, pca, used_features = compute_pca(df, feature_cols=pca_features)
    except ValueError as exc:
        logger.debug("PCA computation failed: %s", exc)
        widget.clear_plot()  # type: ignore[attr-defined]
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            f"{title}<br><span style='color:magenta; font-size:9pt'>{exc}</span>"
        )
        return
    except Exception:
        logger.debug("PCA computation failed", exc_info=True)
        widget.clear_plot()  # type: ignore[attr-defined]
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            f"{title}<br><span style='color:red; font-size:9pt'>"
            "Error computing PCA</span>"
        )
        return

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = (
        pca.explained_variance_ratio_[1] * 100
        if len(pca.explained_variance_ratio_) > 1
        else 0.0
    )

    # Warn when there are fewer samples than features (PCA may be unreliable)
    display_title = title
    n_samples = len(df)
    n_features = len(used_features)
    if n_samples < n_features:
        display_title += "<br><span style='color:magenta; font-size:9pt'>"
        display_title += f"Warning: {n_samples} FOVs < {n_features} features"
        display_title += "</span>"

    _render_scatter(
        widget=widget,
        coords=coords,
        conditions=df["condition"].tolist(),
        x_label=f"PC1 ({var1:.1f}% var)",
        y_label=f"PC2 ({var2:.1f}% var)",
        title=display_title,
    )


# Human-readable short labels for loadings plots (kept brief for axis ticks).
_FEATURE_SHORT_LABELS: dict[str, str] = {
    "mean_amplitude": "Amplitude",
    "mean_frequency": "Frequency",
    "mean_iei": "IEI",
    "mean_spike_freq": "Spike Freq",
    "mean_spike_freq_edges": "Spike Freq (edges)",
    "mean_cell_size": "Cell Size",
    "pct_active": "% Active",
    "burst_count": "Burst Count",
    "burst_avg_duration_s": "Burst Dur.",
    "burst_avg_interval_s": "Burst Int.",
}


def _run_pca_full(
    widget: object,
    engine: Engine,
    run_id: int | None,
    include_stim_status: bool,
) -> tuple[Any, list[str]] | None:
    """Build feature matrix and run PCA, returning (pca, used_features).

    Fits PCA with *all* possible components (not just 2) so that scree and
    loadings plots can show every component.  Returns `None` when PCA
    cannot be computed (too few FOVs, missing data, etc.).
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        df = build_fov_feature_matrix(engine, run_id, include_stim_status)
    except Exception:
        logger.debug("Failed to build FOV feature matrix for PCA", exc_info=True)
        return None

    if df is None or len(df) < 2:
        return None

    pca_features: list[str] | None = getattr(widget, "_pca_features", None)

    try:
        from sklearn.decomposition import PCA

        X, used_features = _prepare_feature_matrix(df, pca_features)
        n_comp = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        return pca, used_features
    except Exception:
        logger.debug("PCA computation failed", exc_info=True)
        return None


def _render_loadings_bar(
    widget: object,
    pca: Any,
    used_features: list[str],
    pc_index: int,
    title: str,
) -> None:
    """Render a horizontal bar chart of PCA loadings for one component."""
    import pyqtgraph as pg

    plot_item = widget.plot_item  # type: ignore[attr-defined]

    loadings = pca.components_[pc_index]
    var_pct = pca.explained_variance_ratio_[pc_index] * 100

    labels = [_FEATURE_SHORT_LABELS.get(f, f) for f in used_features]
    n = len(labels)
    y_positions = np.arange(n)

    # Colour positive loadings blue, negative red
    colors = [
        pg.mkBrush("cornflowerblue") if v >= 0 else pg.mkBrush("tomato")
        for v in loadings
    ]

    bar = pg.BarGraphItem(
        x0=np.zeros(n),
        x1=loadings,
        y=y_positions,
        height=0.6,
        brushes=colors,
        pens=[pg.mkPen("k", width=0.5)] * n,
    )
    plot_item.addItem(bar)

    # Y-axis tick labels
    left_axis = plot_item.getAxis("left")
    left_axis.setTicks([list(zip(y_positions, labels))])

    plot_item.setLabel("bottom", "Loading")
    plot_item.setTitle(f"{title}<br>PC{pc_index + 1} ({var_pct:.1f}% var)")
    plot_item.showGrid(x=True, y=False, alpha=0.3)

    # Add zero line
    from qtpy.QtCore import Qt

    zero_line = pg.InfiniteLine(
        pos=0, angle=90, pen=pg.mkPen("gray", style=Qt.PenStyle.DashLine)
    )
    plot_item.addItem(zero_line)


def _render_scree(
    widget: object,
    pca: Any,
    title: str,
) -> None:
    """Render a scree plot (explained variance per component)."""
    import pyqtgraph as pg

    plot_item = widget.plot_item  # type: ignore[attr-defined]

    var_ratios = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_ratios)
    n = len(var_ratios)
    x = np.arange(1, n + 1)

    # Bar chart for individual variance
    bar = pg.BarGraphItem(
        x=x,
        height=var_ratios,
        width=0.6,
        brush=pg.mkBrush("cornflowerblue"),
        pen=pg.mkPen("k", width=0.5),
    )
    plot_item.addItem(bar)

    # Cumulative line
    cum_line = pg.PlotDataItem(
        x=x,
        y=cum_var,
        pen=pg.mkPen("tomato", width=2),
        symbol="o",
        symbolSize=7,
        symbolBrush=pg.mkBrush("tomato"),
    )
    plot_item.addItem(cum_line)

    # Legend
    legend = plot_item.addLegend(offset=(-10, 5))
    legend.clear()
    legend.addItem(bar, "Individual")
    legend.addItem(cum_line, "Cumulative")

    # X-axis tick labels: PC1, PC2, ...
    bottom_axis = plot_item.getAxis("bottom")
    bottom_axis.setTicks([[(i, f"PC{i}") for i in range(1, n + 1)]])

    plot_item.setLabel("bottom", "Component")
    plot_item.setLabel("left", "Explained Variance (%)")
    plot_item.setTitle(title)
    plot_item.showGrid(x=True, y=True, alpha=0.3)


def plot_pca_loadings(
    widget: object,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot PC1 loadings as a horizontal bar chart.

    Each bar represents one feature; length is the PC1 loading coefficient.
    Positive loadings are blue, negative loadings are red.
    """
    widget.clear_plot()  # type: ignore[attr-defined]
    result = _run_pca_full(widget, engine, run_id, include_stim_status=False)
    if result is None:
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            "PCA Loadings<br><span style='color:magenta; font-size:9pt'>"
            "Need ≥ 2 FOVs for PCA</span>"
        )
        return
    pca, used_features = result
    _render_loadings_bar(widget, pca, used_features, pc_index=0, title="PCA Loadings")


def plot_pca_scree(
    widget: object,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot a scree chart showing explained variance per principal component.

    Bars show individual variance; the red line shows cumulative variance.
    """
    widget.clear_plot()  # type: ignore[attr-defined]
    result = _run_pca_full(widget, engine, run_id, include_stim_status=False)
    if result is None:
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            "PCA Scree Plot<br><span style='color:magenta; font-size:9pt'>"
            "Need ≥ 2 FOVs for PCA</span>"
        )
        return
    pca, _ = result
    _render_scree(widget, pca, title="PCA Scree Plot")


def plot_pca_scatter(
    widget: object,  # _MultilWellGraphWidget
    text: str,
    engine: Engine,
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
        Filter to a single CaliResult; `None` uses all runs in the DB.
    """
    _run_pca_scatter(
        widget=widget,
        engine=engine,
        run_id=run_id,
        include_stim_status=False,
        title="PCA — FOV Feature Space",
    )


def plot_pca_scatter_stim_split(
    widget: object,  # _MultilWellGraphWidget
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot a PCA scatter of FOVs coloured by condition + stimulation status.

    Like :func:`plot_pca_scatter` but each FOV is split into two points —
    one for its stimulated ROIs and one for its non-stimulated ROIs — so that
    the stim/non-stim separation is visible directly in the PCA space.
    Stimulated points are coloured green, non-stimulated points are coloured
    magenta (matching the stim-split bar plots).

    Only meaningful for `Evoked Activity` runs whose ROIs have a
    `stimulated` attribute set.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        Target plot widget.
    text : str
        Plot name (used in the title).
    engine : Engine
        Database engine.
    run_id : int | None
        Filter to a single CaliResult; `None` uses all runs in the DB.
    """
    _run_pca_scatter(
        widget=widget,
        engine=engine,
        run_id=run_id,
        include_stim_status=True,
        title="PCA — FOV Feature Space (Stim vs NonStim)",
    )
