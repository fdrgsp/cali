from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import correlate
from scipy.stats import zscore
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from cali.plot._hover_utils import setup_pick_click_for_heatmap
from cali.sqlmodel._model import FOV, ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.image import AxesImage
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget

from cali.logger import cali_logger


def _get_traces_for_run(roi_model: ROI, run_id: int | None) -> Traces | None:
    """Get the Traces object for a specific run from the ROI's traces_history."""
    if not roi_model.traces_history:
        return None
    if run_id is None:
        return roi_model.traces_history[0] if roi_model.traces_history else None
    for trace in roi_model.traces_history:
        if trace.analysis_result_id == run_id:
            return trace
    return None


def _get_data_analysis_for_run(
    roi_model: ROI, run_id: int | None
) -> DataAnalysis | None:
    """Get DataAnalysis for a specific run from ROI's data_analysis_history."""
    if not roi_model.data_analysis_history:
        return None
    if run_id is None:
        return (
            roi_model.data_analysis_history[0]
            if roi_model.data_analysis_history
            else None
        )
    # First try to find exact match
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return (
        roi_model.data_analysis_history[0] if roi_model.data_analysis_history else None
    )


def _calculate_cross_correlation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for the active ROIs."""
    with Session(engine) as session:
        roi_data = []  # List of (ROI, Traces)

        if run_id is not None:
            # Optimized query
            stmt = (
                select(ROI, Traces)
                .join(FOV, ROI.fov_id == FOV.id)
                .join(
                    Traces,
                    (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
                )
                .where(col(FOV.name) == fov_name)
                .where(col(ROI.active) == True)  # noqa: E712
            )

            if rois is not None:
                stmt = stmt.where(col(ROI.id).in_(rois))

            results = session.exec(stmt).all()
            roi_data = results
        else:
            # Legacy behavior
            # Get detection_settings_id from the run if run_id is provided
            detection_settings_id: int | None = None
            if run_id is not None:
                result = session.get(CaliResult, run_id)
                if result:
                    detection_settings_id = result.detection_settings_id

            stmt = select(ROI).join(FOV).where(col(FOV.name) == fov_name)
            if rois is not None:
                stmt = stmt.where(col(ROI.id).in_(rois))
            # Filter by detection settings if we have a run_id
            if detection_settings_id is not None:
                stmt = stmt.where(
                    col(ROI.detection_settings_id) == detection_settings_id
                )
            stmt = stmt.where(col(ROI.active) == True).options(  # noqa: E712
                selectinload(ROI.traces_history),
            )
            roi_results = session.exec(stmt).all()

            for r in roi_results:
                t = _get_traces_for_run(r, None)
                if t:
                    roi_data.append((r, t))

    traces: list[list[float]] = []
    rois_idxs: list[int] = []

    for roi, roi_traces in roi_data:
        if roi_traces is None or roi_traces.dec_dff is None or roi.label_value is None:
            continue
        rois_idxs.append(roi.label_value)
        traces.append(roi_traces.dec_dff)

    if len(rois_idxs) <= 1:
        cali_logger.warning(
            "Not enough active ROIs to calculate cross-correlation. "
            "At least two active ROIs are required."
        )
        return None, None

    traces_array = np.array(traces)  # shape (n_rois, n_frames)

    dff_zero_mean = zscore(traces_array, axis=1)

    n_rois = len(rois_idxs)
    correlation_matrix_active = np.zeros((n_rois, n_rois))
    for i, j in itertools.product(range(n_rois), range(n_rois)):
        x = dff_zero_mean[i]
        y = dff_zero_mean[j]
        corr = correlate(x, y, mode="full", method="fft")
        corr /= np.linalg.norm(x) * np.linalg.norm(y)  # normalises magnitude
        correlation_matrix_active[i, j] = np.max(corr)
    return correlation_matrix_active, rois_idxs


def _plot_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    correlation_matrix, rois_idxs = _calculate_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        return

    ax.set_title("Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events)")
    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])

    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    img = ax.imshow(correlation_matrix, cmap="viridis", vmin=0, vmax=1, picker=True)

    _add_hover_functionality_cross_corr(img, widget, rois_idxs, correlation_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_cross_corr(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click_for_heatmap(image.axes, widget, rois, values)


# def _plot_hierarchical_clustering_data(
#     widget: _SingleWellGraphWidget,
#     engine: Engine,
#     fov_name: str,
#     rois: list[int] | None = None,
#     run_id: int | None = None,
#     use_dendrogram: bool = False,
# ) -> None:
#     widget.figure.clear()
#     ax = widget.figure.add_subplot(111)

#     correlation_matrix, rois_idxs = _calculate_cross_correlation(
#         engine, fov_name, rois, run_id
#     )

#     if correlation_matrix is None or rois_idxs is None:
#         return

#     if use_dendrogram:
#         _plot_hierarchical_clustering_dendrogram(ax, correlation_matrix, rois_idxs)
#     else:
#         _plot_hierarchical_clustering_map(widget, ax, correlation_matrix, rois_idxs)

#     ax.set_xlabel("ROI")

#     widget.figure.tight_layout()
#     widget.canvas.draw()


# def _plot_hierarchical_clustering_dendrogram(
#     ax: Axes,
#     correlation_matrix: np.ndarray,
#     rois_idxs: list[int],
# ) -> None:
#     """Plot the hierarchical clustering dendrogram."""
#     ax.set_title(
#         "Pairwise Cross-Correlation - Hierarchical Clustering Dendrogram\n"
#         "(Calcium Peaks Events)"
#     )
#     ax.set_ylabel("Distance")
#     correlation_matrix = np.round(correlation_matrix, decimals=8)
#     dist_condensed = squareform(1 - np.abs(correlation_matrix))
#     Z = linkage(dist_condensed, method="complete")
#     labels = [str(i) for i in rois_idxs]
#     dendrogram(Z, ax=ax, labels=labels, leaf_rotation=90, leaf_font_size=12)


# def _plot_hierarchical_clustering_map(
#     widget: _SingleWellGraphWidget,
#     ax: Axes,
#     correlation_matrix: np.ndarray,
#     rois_idxs: list[int],
# ) -> None:
#     """Plot the hierarchical clustering map."""
#     correlation_matrix = np.round(correlation_matrix, decimals=8)
#     dist_condensed = squareform(1 - np.abs(correlation_matrix))
#     order = leaves_list(linkage(dist_condensed, method="complete"))
#     reordered_matrix = correlation_matrix[order][:, order]
#     ax.set_title(
#         "Pairwise Cross-Correlation - Hierarchical Clustering Map\n"
#         "(Calcium Peaks Events)"
#     )
#     ax.set_ylabel("ROI")
#     ax.set_yticklabels([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_xticks([])
#     image = ax.imshow(reordered_matrix, cmap="viridis")

#     cbar = widget.figure.colorbar(
#         cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
#         ax=ax,
#     )
#     cbar.set_label("Cross-Correlation Index")

#     _add_hover_functionality_clustering(
#         image, widget, rois_idxs, order, reordered_matrix
#     )


# def _add_hover_functionality_clustering(
#     image: AxesImage,
#     widget: _SingleWellGraphWidget,
#     rois: list[int],
#     order: list[int],
#     values: np.ndarray,
# ) -> None:
#     """Add hover functionality using mplcursors."""
#     cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

#     @cursor.connect("add")  # type: ignore [misc]
#     def on_add(sel: mplcursors.Selection) -> None:
#         x, y = map(int, np.round(sel.target))
#         roi_x, roi_y = rois[order[x]], rois[order[y]]

#         sel.annotation.set(
#             text=f"ROI {roi_x} ↔ ROI {roi_y}\nvalue: {values[y, x]:0.2f}",
#             fontsize=8,
#             color="black",
#         )

#         widget.roiSelected.emit([str(roi_x), str(roi_y)])
