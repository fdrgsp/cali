from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from cali.plot._hover_utils import setup_pick_hover_for_heatmap
from cali.plot._util import (
    _get_calcium_peaks_event_synchrony,
    _get_calcium_peaks_event_synchrony_matrix,
    _get_calcium_peaks_events_from_rois,
)

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.image import AxesImage
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget

from cali.logger import cali_logger
from cali.sqlmodel._model import ROI, CaliResult, DataAnalysis, Traces


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


def _plot_peak_event_synchrony_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot peak event-based synchrony analysis.

    Parameters
    ----------
    widget: _SingleWellGraphWidget
        widget to plot on
    engine: Engine
        Database engine
    fov_name: str
        Name of the FOV
    rois: list[int] | None
        List of ROI indices to include, None for all
    run_id: int | None
        The run ID to filter by, None for latest
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    peak_trains = _get_calcium_peaks_events_from_rois(engine, fov_name, rois, run_id)
    if peak_trains is None or len(peak_trains) < 2:
        cali_logger.warning(
            "Insufficient peak data for synchrony analysis. "
            "Ensure at least two ROIs with calcium peaks are selected."
        )
        return

    jit = _get_jit(engine, fov_name, rois, run_id)
    if jit is None:
        cali_logger.warning(
            "No valid jitter window value found for synchrony analysis."
        )
        return

    # Convert peak trains to peak event data dict for correlation-based synchrony
    peak_event_data_dict = {
        roi_name: cast("list[float]", peak_train.astype(float).tolist())
        for roi_name, peak_train in peak_trains.items()
    }

    # Use jitter window method for calcium peaks - better suited for discrete
    # events with inherent timing uncertainty due to biology and frame rate limits
    synchrony_matrix = _get_calcium_peaks_event_synchrony_matrix(
        peak_event_data_dict, method="jitter_window", jitter_window=jit
    )

    if synchrony_matrix is None:
        cali_logger.warning(
            "Failed to calculate synchrony matrix. "
            "Ensure peak event data is valid and contains sufficient data."
        )
        return

    # Calculate global synchrony metric using peak event-specific function
    global_synchrony = _get_calcium_peaks_event_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Calcium Peaks Events - Jitter Window Method)\n"
    )

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1, picker=True)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Peak Event Synchrony Index")

    ax.set_title(title)
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel("ROI")
    ax.set_xticklabels([])
    ax.set_xticks([])

    active_rois = list(peak_trains.keys())
    _add_hover_functionality(img, widget, active_rois, synchrony_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()
    # events with inherent timing uncertainty due to biology and frame rate limits
    synchrony_matrix = _get_calcium_peaks_event_synchrony_matrix(
        peak_event_data_dict, method="jitter_window", jitter_window=jit
    )

    if synchrony_matrix is None:
        cali_logger.warning(
            "Failed to calculate synchrony matrix. "
            "Ensure peak event data is valid and contains sufficient data."
        )
        return

    # Calculate global synchrony metric using peak event-specific function
    global_synchrony = _get_calcium_peaks_event_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Calcium Peaks Events - Jitter Window Method)\n"
    )

    ax.set_title(title)
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel("ROI")
    ax.set_xticklabels([])
    ax.set_xticks([])

    active_rois = list(peak_trains.keys())
    _add_hover_functionality(img, widget, active_rois, synchrony_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_jit(
    engine: Engine, fov_name: str, rois: list[int] | None, run_id: int | None = None
) -> int | None:
    """Get the jitter window value for synchrony from database."""
    from sqlmodel import Session, select

    from cali.sqlmodel._model import AnalysisSettings

    with Session(engine) as session:
        # Get the AnalysisSettings from the run
        if run_id is not None:
            result = session.get(CaliResult, run_id)
            if result and result.analysis_settings_id is not None:
                settings = session.get(AnalysisSettings, result.analysis_settings_id)
                if settings:
                    return settings.calcium_sync_jitter_window  # type: ignore[no-any-return]

        # Fallback: get settings from the first available run
        stmt = (
            select(CaliResult)
            .where(CaliResult.analysis_settings_id.is_not(None))  # type: ignore
            .limit(1)
        )
        result = session.exec(stmt).first()
        if result and result.analysis_settings_id is not None:
            settings = session.get(AnalysisSettings, result.analysis_settings_id)
            if settings:
                return settings.calcium_sync_jitter_window  # type: ignore[no-any-return]

    cali_logger.warning("No valid analysis settings found for synchrony analysis.")
    return None


def _add_hover_functionality(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[str],
    synchrony_matrix: np.ndarray,
) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_hover_for_heatmap(image.axes, widget, rois, synchrony_matrix)
