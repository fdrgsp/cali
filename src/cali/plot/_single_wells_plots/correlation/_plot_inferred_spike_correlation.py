from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.signal import correlate
from scipy.spatial.distance import squareform
from scipy.stats import zscore

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click_for_heatmap
from cali.plot._util import _get_data_analysis_for_run, _get_traces_for_run

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Cross-correlation computation
# -----------------------------------------------------------------------------#
def _calculate_spike_cross_correlation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for spike trains from active ROIs.

    Uses thresholded inferred_spikes → binary spike trains, then computes
    maximum normalized cross-correlation over all lags.
    """
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI

    spike_trains: list[np.ndarray] = []
    rois_idxs: list[int] = []

    # Query ROIs from database
    with Session(engine) as session:
        stmt = select(ROI).join(FOV).where(col(FOV.name) == fov_name)
        if rois is not None:
            stmt = stmt.where(col(ROI.id).in_(rois))
        stmt = stmt.where(col(ROI.active) == True).options(  # noqa: E712
            selectinload(ROI.traces_history),
            selectinload(ROI.data_analysis_history),
        )
        roi_results = session.exec(stmt).all()

    # Extract spike trains for the active ROIs
    for roi in roi_results:
        traces = _get_traces_for_run(roi, run_id)
        data_analysis = _get_data_analysis_for_run(roi, run_id)

        if traces is None or data_analysis is None:
            continue

        inferred_spikes = traces.inferred_spikes
        inferred_spikes_threshold = data_analysis.inferred_spikes_threshold

        if inferred_spikes is None or inferred_spikes_threshold is None:
            continue

        spikes = np.asarray(inferred_spikes, dtype=float)
        the = float(inferred_spikes_threshold)

        # Threshold and binarize (vectorized)
        spikes[spikes <= the] = 0.0
        spike_train = (spikes > 0.0).astype(float)

        if spike_train.sum() <= 0:
            # Skip ROIs with no spikes
            continue

        rois_idxs.append(int(roi.label_value))
        spike_trains.append(spike_train)

    if len(rois_idxs) <= 1:
        cali_logger.warning(
            "Insufficient spike data for correlation analysis. "
            "Need at least 2 ROIs with spikes."
        )
        return None, None

    # Convert to array: shape (n_rois, n_frames)
    spike_trains_array = np.vstack(spike_trains)

    # Z-score per ROI (handle varying firing rates)
    spike_trains_zscore = zscore(spike_trains_array, axis=1, nan_policy="omit")
    spike_trains_zscore = np.nan_to_num(
        spike_trains_zscore, nan=0.0, posinf=0.0, neginf=0.0
    )

    n_rois = len(rois_idxs)
    correlation_matrix = np.empty((n_rois, n_rois), dtype=float)

    # Precompute norms, avoid repeated work
    norms = np.linalg.norm(spike_trains_zscore, axis=1)
    norms[norms == 0] = np.finfo(float).eps  # avoid division by zero

    # Diagonal is self-correlation = 1
    np.fill_diagonal(correlation_matrix, 1.0)

    # Compute only upper triangle, mirror to lower
    for i in range(n_rois):
        x = spike_trains_zscore[i]
        for j in range(i + 1, n_rois):
            y = spike_trains_zscore[j]

            # FFT-based cross-correlation over all lags
            corr = correlate(x, y, mode="full", method="fft")
            corr /= norms[i] * norms[j]

            # Use max absolute correlation
            max_corr = float(np.max(np.abs(corr)))
            correlation_matrix[i, j] = max_corr
            correlation_matrix[j, i] = max_corr

    return correlation_matrix, rois_idxs


# -----------------------------------------------------------------------------#
# Heatmap plot
# -----------------------------------------------------------------------------#
def _plot_spike_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot pairwise cross-correlation matrix for spike trains."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        cali_logger.warning(
            "Insufficient spike data for cross-correlation analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    ax.set_title("Pairwise Cross-Correlation Matrix\n(Thresholded Spike Data)")
    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Colorbar
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=norm),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    # Heatmap (picker=True for hover)
    img = ax.imshow(
        correlation_matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        picker=True,
    )

    _add_hover_functionality_spike_corr(img, widget, rois_idxs, correlation_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_spike_corr(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality using efficient pick events for spike correlation."""
    setup_pick_click_for_heatmap(image.axes, widget, rois, values)


# -----------------------------------------------------------------------------#
# Hierarchical clustering plots
# -----------------------------------------------------------------------------#
def _plot_spike_hierarchical_clustering_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    use_dendrogram: bool = False,
) -> None:
    """Plot hierarchical clustering analysis for spike correlation data."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        cali_logger.warning(
            "Insufficient spike data for hierarchical clustering analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    if use_dendrogram:
        _plot_spike_hierarchical_clustering_dendrogram(
            ax, correlation_matrix, rois_idxs
        )
    else:
        _plot_spike_hierarchical_clustering_map(
            widget, ax, correlation_matrix, rois_idxs
        )

    ax.set_xlabel("ROI")
    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_spike_hierarchical_clustering_dendrogram(
    ax: Axes,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering dendrogram for spike correlation data."""
    ax.set_title(
        "Pairwise Cross-Correlation - Hierarchical Clustering Dendrogram\n"
        "(Thresholded Spike Data)"
    )
    ax.set_ylabel("Distance")

    # Stabilize numerics
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Convert correlation to distance (1 - |corr|)
    dist_condensed = squareform(1.0 - np.abs(correlation_matrix))

    # Complete-linkage clustering
    Z = linkage(dist_condensed, method="complete")

    labels = [str(i) for i in rois_idxs]

    dendrogram(Z, ax=ax, labels=labels, leaf_rotation=90, leaf_font_size=12)


def _plot_spike_hierarchical_clustering_map(
    widget: _SingleWellGraphWidget,
    ax: Axes,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering heatmap for spike correlation data."""
    # Stabilize numerics
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Distance → clustering → leaf order
    dist_condensed = squareform(1.0 - np.abs(correlation_matrix))
    order = leaves_list(linkage(dist_condensed, method="complete"))

    # Reorder matrix
    reordered_matrix = correlation_matrix[order][:, order]

    ax.set_title(
        "Pairwise Cross-Correlation - Hierarchical Clustering Map\n"
        "(Thresholded Spike Data)"
    )
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_box_aspect(1)

    img = ax.imshow(
        reordered_matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    # Colorbar
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=norm),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    _add_hover_functionality_spike_clustering(
        img, widget, rois_idxs, order, reordered_matrix
    )


def _add_hover_functionality_spike_clustering(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    order: np.ndarray,
    values: np.ndarray,
) -> None:
    """Add hover functionality for spike clustering heatmap."""
    # Reorder ROI IDs to match reordered_matrix indexing
    reordered_roi_ids = [rois[i] for i in order]
    setup_pick_click_for_heatmap(image.axes, widget, reordered_roi_ids, values)
