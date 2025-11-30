from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.signal import correlate
from scipy.spatial.distance import squareform
from scipy.stats import zscore

from cali.plot._util import _get_data_analysis_for_run, _get_traces_for_run

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget

from cali.logger import cali_logger


def _calculate_spike_cross_correlation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for spike trains from active ROIs.

    This function extracts thresholded spike data from ROIs and computes pairwise
    cross-correlations using the same approach as calcium trace correlation but
    applied to binary spike trains.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of specific ROI indices to analyze, None for all active ROIs
    run_id : int | None
        The run ID to filter by, None for latest

    Returns
    -------
    tuple[np.ndarray | None, list[int] | None]
        Correlation matrix and corresponding ROI indices, or (None, None) if
        insufficient data
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
        # Get traces and data_analysis for the specified run
        traces = _get_traces_for_run(roi, run_id)
        data_analysis = _get_data_analysis_for_run(roi, run_id)

        if traces is None or data_analysis is None:
            continue

        # Get spike data from Traces and threshold from DataAnalysis
        inferred_spikes = traces.inferred_spikes
        inferred_spikes_threshold = data_analysis.inferred_spikes_threshold

        if inferred_spikes is None or inferred_spikes_threshold is None:
            continue

        # Convert spike probabilities to binary spike train
        spike_probs = [
            spike if spike > inferred_spikes_threshold else 0.0
            for spike in inferred_spikes
        ]
        spike_train = np.array(spike_probs) > 0.0

        # Only include ROIs that have at least one spike
        if np.sum(spike_train) > 0:
            rois_idxs.append(roi.id)
            spike_trains.append(spike_train.astype(float))

    if len(rois_idxs) <= 1:
        cali_logger.warning(
            "Insufficient spike data for correlation analysis. "
            "Need at least 2 ROIs with spikes."
        )
        return None, None

    # Convert to array for processing
    spike_trains_array = np.array(spike_trains)  # shape (n_rois, n_frames)

    # Z-score normalization (mean centering and std normalization)
    # This is important for spike trains to handle different firing rates
    spike_trains_zscore = zscore(spike_trains_array, axis=1, nan_policy="omit")

    # Handle cases where std is 0 (constant spike trains)
    spike_trains_zscore = np.nan_to_num(
        spike_trains_zscore, nan=0.0, posinf=0.0, neginf=0.0
    )

    n_rois = len(rois_idxs)
    correlation_matrix = np.zeros((n_rois, n_rois))

    # Calculate pairwise cross-correlations
    for i, j in itertools.product(range(n_rois), range(n_rois)):
        x = spike_trains_zscore[i]
        y = spike_trains_zscore[j]

        # Compute cross-correlation using FFT method for efficiency
        corr = correlate(x, y, mode="full", method="fft")

        # Normalize by the norms of the signals
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x > 0 and norm_y > 0:
            corr /= norm_x * norm_y
            correlation_matrix[i, j] = np.max(np.abs(corr))  # Take max absolute value
        else:
            correlation_matrix[i, j] = 0.0  # No correlation for constant signals

    return correlation_matrix, rois_idxs


def _plot_spike_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot pairwise cross-correlation matrix for spike trains.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    run_id : int | None
        The run ID to filter by, None for latest
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        cali_logger.warning(
            "Insufficient spike data for cross-correlation analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        return

    ax.set_title("Pairwise Cross-Correlation Matrix\n(Thresholded Spike Data)")
    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Create colorbar
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    # Display the correlation matrix
    img = ax.imshow(correlation_matrix, cmap="viridis", vmin=0, vmax=1)

    # Add hover functionality
    _add_hover_functionality_spike_corr(img, widget, rois_idxs, correlation_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_spike_corr(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality using efficient pick events for spike correlation.

    Parameters
    ----------
    image : AxesImage
        The imshow image object
    widget : _SingleWellGraphWidget
        Widget containing the plot
    rois : list[int]
        List of ROI indices
    values : np.ndarray
        Correlation matrix values
    """
    from cali.plot._hover_utils import setup_pick_hover_for_heatmap

    # Convert ROI indices to strings for hover utils
    roi_strs = [str(r) for r in rois]
    setup_pick_hover_for_heatmap(image.axes, widget, roi_strs, values)


def _plot_spike_hierarchical_clustering_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    use_dendrogram: bool = False,
) -> None:
    """Plot hierarchical clustering analysis for spike correlation data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    run_id : int | None
        The run ID to filter by, None for latest
    use_dendrogram : bool
        If True, plot dendrogram; if False, plot clustered heatmap
    """
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
    """Plot the hierarchical clustering dendrogram for spike correlation data.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    correlation_matrix : np.ndarray
        Correlation matrix
    rois_idxs : list[int]
        List of ROI indices
    """
    ax.set_title(
        ""
        "Pairwise Cross-Correlation - Hierarchical Clustering Dendrogram\n"
        "(Thresholded Spike Data)"
    )
    ax.set_ylabel("Distance")

    # Round to avoid numerical precision issues
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Convert correlation to distance (1 - correlation)
    dist_condensed = squareform(1 - np.abs(correlation_matrix))

    # Perform hierarchical clustering
    Z = linkage(dist_condensed, method="complete")

    # Create labels
    labels = [str(i) for i in rois_idxs]

    # Plot dendrogram
    dendrogram(Z, ax=ax, labels=labels, leaf_rotation=90, leaf_font_size=12)


def _plot_spike_hierarchical_clustering_map(
    widget: _SingleWellGraphWidget,
    ax: Axes,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering map for spike correlation data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget containing the plot
    ax : Axes
        Matplotlib axes to plot on
    correlation_matrix : np.ndarray
        Correlation matrix
    rois_idxs : list[int]
        List of ROI indices
    """
    # Round to avoid numerical precision issues
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Convert correlation to distance and perform clustering
    dist_condensed = squareform(1 - np.abs(correlation_matrix))
    order = leaves_list(linkage(dist_condensed, method="complete"))

    # Reorder matrix according to clustering
    reordered_matrix = correlation_matrix[order][:, order]

    ax.set_title(
        "Pairwise Cross-Correlation - Hierarchical Clustering Map\n"
        "(Thresholded Spike Data)"
    )
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_box_aspect(1)

    # Display the reordered correlation matrix
    image = ax.imshow(reordered_matrix, cmap="viridis", vmin=0, vmax=1)

    # Add colorbar
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    # Add hover functionality
    _add_hover_functionality_spike_clustering(
        image, widget, rois_idxs, order, reordered_matrix
    )


def _add_hover_functionality_spike_clustering(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    order: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality for spike clustering heatmap.

    Parameters
    ----------
    image : AxesImage
        The imshow image object
    widget : _SingleWellGraphWidget
        Widget containing the plot
    rois : list[int]
        List of ROI indices
    order : list[int]
        Clustering order indices
    values : np.ndarray
        Reordered correlation matrix values
    """
    from cali.plot._hover_utils import setup_pick_hover_for_heatmap

    # Create reordered ROI list for hover display
    reordered_rois = [str(rois[i]) for i in order]
    setup_pick_hover_for_heatmap(image.axes, widget, reordered_rois, values)
