"""Dimensionality reduction utilities for multi-well per-condition analysis.

This module builds a per-FOV feature matrix from scalar calcium imaging metrics
and applies PCA for visualisation.

The feature matrix has one row per FOV and includes:

- Per-ROI scalars averaged to FOV level (amplitude, frequency, IEI, spike freq,
  cell size, % active).
- FOVAnalysis burst stats (burst count, avg duration, avg interval).

Usage
-----
>>> from cali.sqlmodel import create_cali_engine
>>> engine = create_cali_engine("sqlite:///my.cali")
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

from cali.plot._spike_data import get_stored_spike_capabilities, roi_is_active
from cali.plot._spike_fov_data import selected_spike_fov
from cali.sqlmodel._engine import ensure_schema_current
from cali.sqlmodel._spike_settings import canonical_spike_methods

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.engine import Engine

    from cali.sqlmodel._spike_settings import SpikeMethod

# OASIS keeps historical column IDs, with explicit scientific labels/metadata.
_CALCIUM_FEATURES = [
    "mean_amplitude",
    "mean_frequency",
    "mean_iei",
    "mean_cell_size",
    "pct_active",
]
_BURST_FEATURES = ["burst_count", "burst_avg_duration_s", "burst_avg_interval_s"]
_OASIS_SPIKE_FEATURES = {
    "mean_spike_freq": "suprathreshold_sample_rate_hz",
    "mean_spike_freq_edges": "suprathreshold_rising_edge_rate_hz",
}
_CASCADE_SPIKE_FEATURES = {
    "mean_expected_spike_rate_hz": "expected_spike_rate_hz",
    "mean_expected_spike_count": "expected_spike_count",
    "mean_suprathreshold_excursion_rate_hz": "suprathreshold_excursion_rate_hz",
}
FEATURE_COLUMNS = [
    "mean_amplitude",
    "mean_frequency",
    "mean_iei",
    *_OASIS_SPIKE_FEATURES,
    "mean_cell_size",
    "pct_active",
    "pct_spike_active",
    *_BURST_FEATURES,
]
CASCADE_FEATURE_COLUMNS = [
    *_CALCIUM_FEATURES,
    *_CASCADE_SPIKE_FEATURES,
    "pct_spike_active",
    *_BURST_FEATURES,
]


def spike_feature_columns(method: SpikeMethod) -> list[str]:
    """Return the selected method's scientifically named features."""
    canonical_spike_methods((method,))
    return list(FEATURE_COLUMNS if method == "oasis" else CASCADE_FEATURE_COLUMNS)


def build_fov_feature_matrix(
    engine: Engine,
    run_id: int | None = None,
    include_stim_status: bool = False,
    *,
    spike_method: SpikeMethod = "oasis",
) -> pd.DataFrame:
    """Build one run's FOV features using independent calcium/spike populations.

    Parameters
    ----------
    engine : Engine
        Database engine.
    run_id : int | None
        Selected analysis run; required when more than one analyzed run exists.
    include_stim_status : bool
        Split ROI metrics by stimulation status; omit shared FOV burst metrics.
    spike_method : {"oasis", "cascade"}
        Method of the stored spike metrics; calcium metrics remain shared.

    Returns
    -------
    pandas.DataFrame
        FOV/condition identifiers and scientifically named feature columns, with
        the selected method, run, feature IDs and meanings recorded in attrs.
    """
    import pandas as pd
    from sqlmodel import Session, col, select

    from cali.sqlmodel import FOV, ROI, DataAnalysis, FOVAnalysis, Well

    from ._util import _get_condition_label, _get_experiment_type

    canonical_spike_methods((spike_method,))
    columns = spike_feature_columns(spike_method)
    spike_fields = (
        _OASIS_SPIKE_FEATURES if spike_method == "oasis" else _CASCADE_SPIKE_FEATURES
    )
    rows: list[dict] = []
    ensure_schema_current(engine)
    with Session(engine) as session:
        if run_id is None:
            ids = set(session.exec(select(DataAnalysis.analysis_result_id)).all())
            ids.update(session.exec(select(FOVAnalysis.analysis_result_id)).all())
            ids.discard(None)
            if len(ids) > 1:
                raise ValueError("PCA requires one analysis run; select run_id.")
            run_id = next(iter(ids)) if ids else None
        if run_id is not None:
            available = get_stored_spike_capabilities(engine, run_id)
            if (available and spike_method not in available) or (
                not available and spike_method == "cascade"
            ):
                raise ValueError(f"No stored {spike_method} results for PCA.")
        experiment_type = (
            _get_experiment_type(session, run_id)
            if include_stim_status and run_id is not None
            else None
        )
        stmt = (
            select(DataAnalysis, ROI, FOV, Well)
            .join(ROI, DataAnalysis.roi_id == ROI.id)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
        )
        if run_id is not None:
            stmt = stmt.where(col(DataAnalysis.analysis_result_id) == run_id)
        FovKey: TypeAlias = tuple
        values: dict[FovKey, dict[str, list[float]]] = {}
        metadata: dict[FovKey, tuple[str, str]] = {}
        counts: dict[FovKey, list[int]] = {}
        seen = set()
        for analysis, roi, fov, well in session.exec(stmt).all():
            key = (fov.id, roi.stimulated) if include_stim_status else (fov.id,)
            if (analysis.analysis_result_id, roi.id) in seen:
                raise ValueError("PCA cannot pool duplicate ROI analyses.")
            seen.add((analysis.analysis_result_id, roi.id))
            suffix = (
                ("_stim" if roi.stimulated else "_non_stim")
                if include_stim_status and roi.stimulated is not None
                else ""
            )
            metadata[key] = (
                fov.name + suffix,
                _get_condition_label(
                    well, roi if include_stim_status else None, experiment_type
                ),
            )
            bucket = values.setdefault(key, {column: [] for column in columns})
            child = analysis.get_spike_analysis(spike_method)
            total = counts.setdefault(key, [0, 0, 0, 0])
            total[0] += 1
            calcium_active = roi_is_active(roi, analysis)
            spike_active = roi_is_active(roi, analysis, spike_method)
            total[1] += int(calcium_active)
            total[2] += int(spike_active)
            total[3] += int(child is not None)
            if calcium_active:
                if analysis.peaks_amplitudes_den_dff:
                    bucket["mean_amplitude"].append(
                        float(np.mean(analysis.peaks_amplitudes_den_dff))
                    )
                if analysis.den_dff_frequency is not None:
                    bucket["mean_frequency"].append(float(analysis.den_dff_frequency))
                if analysis.iei:
                    bucket["mean_iei"].append(float(np.mean(analysis.iei)))
                if roi.cell_size is not None:
                    bucket["mean_cell_size"].append(float(roi.cell_size))
            if child is not None:
                if child.units != (
                    "a.u." if spike_method == "oasis" else "spikes/frame"
                ):
                    raise ValueError(
                        "PCA spike metric units must match the selected method."
                    )
                if spike_active:
                    for column, field in spike_fields.items():
                        value = getattr(child, field)
                        if value is not None:
                            bucket[column].append(float(value))
        fov_stmt = (
            select(FOVAnalysis, FOV, Well)
            .join(FOV, FOVAnalysis.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
        )
        if run_id is not None:
            fov_stmt = fov_stmt.where(col(FOVAnalysis.analysis_result_id) == run_id)
        parents = {}
        for parent, fov, well in session.exec(fov_stmt).all():
            if fov.id in parents:
                raise ValueError("PCA cannot pool multiple FOV analysis runs.")
            parents[fov.id] = selected_spike_fov(parent, spike_method)
            if not include_stim_status:
                key = (fov.id,)
                metadata.setdefault(key, (fov.name, _get_condition_label(well)))
                values.setdefault(key, {column: [] for column in columns})
        for key in sorted(values, key=lambda key: (key[0], str(key[1:]))):
            fov_name, condition = metadata[key]
            row: dict[str, object] = {"fov_name": fov_name, "condition": condition}
            for column, observations in values[key].items():
                row[column] = (
                    float(np.mean(observations)) if observations else float("nan")
                )
            roi_count, calcium, spikes, observed = counts.get(key, [0, 0, 0, 0])
            row["pct_active"] = calcium * 100 / roi_count if roi_count else float("nan")
            row["pct_spike_active"] = (
                spikes * 100 / observed if observed else float("nan")
            )
            child = parents.get(key[0])
            if not include_stim_status and child is not None:
                for column, field in zip(
                    _BURST_FEATURES,
                    (
                        "spike_burst_count",
                        "spike_burst_avg_duration",
                        "spike_burst_avg_interval",
                    ),
                ):
                    value = getattr(child, field)
                    row[column] = float(value) if value is not None else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows, columns=["fov_name", "condition", *columns])
    df.attrs.update(
        spike_method=spike_method,
        run_id=run_id,
        feature_columns=columns,
        spike_metric_fields=dict(spike_fields),
        calcium_activity_field="calcium_active",
        spike_activity_field="spike_active",
    )
    return df


def _selected_pca_features(
    df: pd.DataFrame, feature_cols: list[str] | None
) -> list[str]:
    """Reject mixed methods or incompatible feature IDs before computing PCA."""
    method = df.attrs.get("spike_method")
    if method is not None:
        allowed = spike_feature_columns(method)
    else:
        oasis = any(c in df and not df[c].isna().all() for c in _OASIS_SPIKE_FEATURES)
        cascade = any(
            c in df and not df[c].isna().all() for c in _CASCADE_SPIKE_FEATURES
        )
        if oasis and cascade:
            raise ValueError("PCA cannot pool OASIS and CASCADE spike metrics.")
        allowed = list(CASCADE_FEATURE_COLUMNS if cascade else FEATURE_COLUMNS)
        named_spike_features = set(_OASIS_SPIKE_FEATURES) | set(_CASCADE_SPIKE_FEATURES)
        allowed.extend(c for c in df.columns if c not in named_spike_features)
    other = _OASIS_SPIKE_FEATURES if method == "cascade" else _CASCADE_SPIKE_FEATURES
    if method is not None and any(c in df and not df[c].isna().all() for c in other):
        raise ValueError("PCA cannot pool OASIS and CASCADE spike metrics.")
    if feature_cols is None:
        feature_cols = [c for c in allowed if c in df.columns]
    if set(feature_cols) - set(allowed) or set(feature_cols) - set(df.columns):
        raise ValueError("PCA features must belong to the selected spike method.")

    return feature_cols


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

    feature_cols = _selected_pca_features(df, feature_cols)

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
        y = coords[mask, 1] if coords.shape[1] > 1 else np.zeros(int(mask.sum()))
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
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Shared implementation for PCA scatter plots."""
    canonical_spike_methods((spike_method,))
    import logging

    logger = logging.getLogger(__name__)

    try:
        df = build_fov_feature_matrix(
            engine, run_id, include_stim_status, spike_method=spike_method
        )
    except ValueError:
        raise
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
    pca_features = _selected_pca_features(df, getattr(widget, "_pca_features", None))

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
        y_label=f"PC2 ({var2:.1f}% var)"
        if len(pca.explained_variance_ratio_) > 1
        else "PC2 (unavailable)",
        title=display_title,
    )


# Human-readable short labels for loadings plots (kept brief for axis ticks).
_FEATURE_SHORT_LABELS: dict[str, str] = {
    "mean_amplitude": "Amplitude",
    "mean_frequency": "Frequency",
    "mean_iei": "IEI",
    "mean_spike_freq": "Samples Above Cutoff (Hz)",
    "mean_spike_freq_edges": "Rising Edges (Hz)",
    "mean_cell_size": "Cell Size",
    "pct_active": "% Calcium Active",
    "pct_spike_active": "% Spike Active",
    "mean_expected_spike_rate_hz": "Expected Rate (Hz)",
    "mean_expected_spike_count": "Expected Count",
    "mean_suprathreshold_excursion_rate_hz": "Excursions (Hz)",
    "burst_count": "Burst Count",
    "burst_avg_duration_s": "Burst Dur.",
    "burst_avg_interval_s": "Burst Int.",
}


def _run_pca_full(
    widget: object,
    engine: Engine,
    run_id: int | None,
    include_stim_status: bool,
    *,
    spike_method: SpikeMethod = "oasis",
) -> tuple[Any, list[str]] | None:
    """Build feature matrix and run PCA, returning (pca, used_features).

    Fits PCA with *all* possible components (not just 2) so that scree and
    loadings plots can show every component.  Returns `None` when PCA
    cannot be computed (too few FOVs, missing data, etc.).
    """
    canonical_spike_methods((spike_method,))
    import logging

    logger = logging.getLogger(__name__)

    try:
        df = build_fov_feature_matrix(
            engine, run_id, include_stim_status, spike_method=spike_method
        )
    except ValueError:
        raise
    except Exception:
        logger.debug("Failed to build FOV feature matrix for PCA", exc_info=True)
        return None

    if df is None or len(df) < 2:
        return None

    pca_features = _selected_pca_features(df, getattr(widget, "_pca_features", None))

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
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot PC1 loadings as a horizontal bar chart.

    Each bar represents one feature; length is the PC1 loading coefficient.
    Positive loadings are blue, negative loadings are red.
    """
    canonical_spike_methods((spike_method,))
    widget.clear_plot()  # type: ignore[attr-defined]
    result = _run_pca_full(
        widget, engine, run_id, include_stim_status=False, spike_method=spike_method
    )
    if result is None:
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            f"[{spike_method.upper()}] PCA Loadings<br>"
            "<span style='color:magenta; font-size:9pt'>"
            "Need ≥ 2 FOVs for PCA</span>"
        )
        return
    pca, used_features = result
    _render_loadings_bar(
        widget,
        pca,
        used_features,
        pc_index=0,
        title=f"[{spike_method.upper()}] PCA Loadings",
    )


def plot_pca_scree(
    widget: object,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Plot a scree chart showing explained variance per principal component.

    Bars show individual variance; the red line shows cumulative variance.
    """
    canonical_spike_methods((spike_method,))
    widget.clear_plot()  # type: ignore[attr-defined]
    result = _run_pca_full(
        widget, engine, run_id, include_stim_status=False, spike_method=spike_method
    )
    if result is None:
        widget.plot_item.setTitle(  # type: ignore[attr-defined]
            f"[{spike_method.upper()}] PCA Scree Plot<br>"
            "<span style='color:magenta; font-size:9pt'>"
            "Need ≥ 2 FOVs for PCA</span>"
        )
        return
    pca, _ = result
    _render_scree(widget, pca, title=f"[{spike_method.upper()}] PCA Scree Plot")


def plot_pca_scatter(
    widget: object,  # _MultilWellGraphWidget
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
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
    spike_method : {"oasis", "cascade"}
        Method of the stored spike feature values.
    run_id : int | None
        Filter to a single CaliResult; `None` uses all runs in the DB.
    """
    canonical_spike_methods((spike_method,))
    _run_pca_scatter(
        widget=widget,
        engine=engine,
        run_id=run_id,
        include_stim_status=False,
        spike_method=spike_method,
        title=f"[{spike_method.upper()}] PCA — FOV Feature Space",
    )


def plot_pca_scatter_stim_split(
    widget: object,  # _MultilWellGraphWidget
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod = "oasis",
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
    spike_method : {"oasis", "cascade"}
        Method of the stored spike feature values.
    run_id : int | None
        Filter to a single CaliResult; `None` uses all runs in the DB.
    """
    canonical_spike_methods((spike_method,))
    _run_pca_scatter(
        widget=widget,
        engine=engine,
        run_id=run_id,
        include_stim_status=True,
        spike_method=spike_method,
        title=f"[{spike_method.upper()}] PCA — FOV Feature Space (Stim vs NonStim)",
    )
