from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import correlate
from scipy.stats import zscore
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click_for_heatmap
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.image import AxesImage
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Helpers: retrieval from ROI histories
# -----------------------------------------------------------------------------#
def _get_traces_for_run(roi_model: ROI, run_id: int | None) -> Traces | None:
    """Get the Traces object for a specific run from the ROI's traces_history."""
    if not roi_model.traces_history:
        return None
    if run_id is None:
        return roi_model.traces_history[0]
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
        return roi_model.data_analysis_history[0]
    # First try to find exact match
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return roi_model.data_analysis_history[0]


# -----------------------------------------------------------------------------#
# Cross-correlation computation
# -----------------------------------------------------------------------------#
def _calculate_cross_correlation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for the active ROIs.

    The value stored is the **maximum** normalized cross-correlation over all lags.
    """
    with Session(engine) as session:
        roi_data: list[tuple[ROI, Traces]] = []

        if run_id is None:
            cali_logger.warning("No run ID specified for cross-correlation plot.")
            return None, None

        # Preferred: direct join on the selected run
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

        roi_data = session.exec(stmt).all()

    traces: list[np.ndarray] = []
    rois_idxs: list[int] = []

    for roi, roi_traces in roi_data:
        if roi_traces is None or roi_traces.dec_dff is None or roi.label_value is None:
            continue

        tr = np.asarray(roi_traces.dec_dff, dtype=float)
        if tr.ndim != 1 or tr.size == 0:
            continue

        rois_idxs.append(int(roi.label_value))
        traces.append(tr)

    if len(rois_idxs) <= 1:
        cali_logger.warning(
            "Not enough active ROIs to calculate cross-correlation. "
            "At least two active ROIs are required."
        )
        return None, None

    # Stack into array: shape (n_rois, n_frames)
    traces_array = np.vstack(traces)  # assumes all same length (as in Ca imaging)

    # Z-score along time (axis=1) -> zero-mean, unit variance per ROI
    dff_zero_mean = zscore(traces_array, axis=1)

    n_rois = len(rois_idxs)
    correlation_matrix_active = np.empty((n_rois, n_rois), dtype=float)

    # Precompute norms to avoid repeated work
    norms = np.linalg.norm(dff_zero_mean, axis=1)
    # Guard against zero norms (flat traces)
    norms[norms == 0] = np.finfo(float).eps

    # Diagonal is always 1 (self-correlation)
    np.fill_diagonal(correlation_matrix_active, 1.0)

    # Only compute i < j, then mirror to j < i
    for i in range(n_rois):
        x = dff_zero_mean[i]
        for j in range(i + 1, n_rois):
            y = dff_zero_mean[j]
            # full cross-correlation over lags
            corr = correlate(x, y, mode="full", method="fft")
            corr /= norms[i] * norms[j]
            max_corr = float(np.max(corr))
            correlation_matrix_active[i, j] = max_corr
            correlation_matrix_active[j, i] = max_corr

    return correlation_matrix_active, rois_idxs


# -----------------------------------------------------------------------------#
# Plotting
# -----------------------------------------------------------------------------#
def _plot_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot the pairwise cross-correlation matrix as a heatmap."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    correlation_matrix, rois_idxs = _calculate_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    ax.set_title("Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events)")
    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Colorbar with fixed [0, 1] range
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=norm),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    img = ax.imshow(
        correlation_matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        picker=True,
    )

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
