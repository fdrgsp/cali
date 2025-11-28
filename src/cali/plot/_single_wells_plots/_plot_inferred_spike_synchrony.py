from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors
import numpy as np

from cali.plot._util import (
    _get_data_analysis_for_run,
    _get_spike_synchrony,
    _get_spike_synchrony_matrix,
    _get_traces_for_run,
)

if TYPE_CHECKING:
    from matplotlib.image import AxesImage
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget

from cali.logger import cali_logger


def _plot_spike_synchrony_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot spike-based synchrony analysis.

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

    spike_trains = _get_spike_trains_from_rois(engine, fov_name, rois, run_id)
    if spike_trains is None or len(spike_trains) < 2:
        cali_logger.warning(
            "Insufficient spike data for synchrony analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        return

    lag = _get_lag(engine, fov_name, rois, run_id)
    if lag is None:
        cali_logger.warning("No valid lag value found for synchrony analysis.")
        return

    # Convert spike trains to spike data dict for correlation-based synchrony
    spike_data_dict = {
        roi_name: cast("list[float]", spike_train.astype(float).tolist())
        for roi_name, spike_train in spike_trains.items()
    }

    # Use cross-correlation method for inferred spikes - better suited for
    # signal-like data that may have temporal artifacts from deconvolution
    synchrony_matrix = _get_spike_synchrony_matrix(
        spike_data_dict, method="cross_correlation", max_lag=lag
    )

    if synchrony_matrix is None:
        cali_logger.warning(
            "Failed to compute synchrony matrix. "
            "Ensure spike data is valid and contains sufficient ROIs."
        )
        return

    # Calculate global synchrony metric using spike-specific function
    global_synchrony = _get_spike_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Thresholded Spike Data - Cross-Correlation Method)\n"
    )

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Spike Synchrony Index")

    ax.set_title(title)
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel("ROI")
    ax.set_xticklabels([])
    ax.set_xticks([])

    active_rois = list(spike_trains.keys())
    _add_hover_functionality(img, widget, active_rois, synchrony_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_lag(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> int | None:
    """Get the lag value for synchrony from AnalysisSettings."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, CaliResult, Experiment, Plate, Well

    with Session(engine) as session:
        # Get CaliResult for this run via FOV -> Well -> Plate -> Experiment
        stmt = (
            select(CaliResult)
            .join(Experiment, CaliResult.experiment == Experiment.id)
            .join(Plate, Experiment.id == Plate.experiment_id)
            .join(Well, Plate.id == Well.plate_id)
            .join(FOV, Well.id == FOV.well_id)
            .where(col(FOV.name) == fov_name)
        )
        if run_id is not None:
            stmt = stmt.where(col(CaliResult.id) == run_id)
        result = session.exec(stmt).first()

        if result is None or result.analysis_settings is None:
            cali_logger.warning("No analysis settings found for synchrony analysis.")
            return None

        # Get spike synchrony cross-correlation lag from analysis settings
        lag = result.analysis_settings.spikes_sync_cross_corr_lag
        return lag if lag is not None else 5  # Default value


def _get_spike_trains_from_rois(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> dict[str, np.ndarray] | None:
    """Extract spike trains from ROI data.

    Args:
        engine: Database engine
        fov_name: Name of the FOV
        rois: List of ROI indices to include, None for all
        run_id: The run ID to filter by, None for latest

    Returns
    -------
        Dictionary mapping ROI names to binary spike arrays
    """
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI

    spike_trains: dict[str, np.ndarray] = {}

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

    if len(roi_results) < 2:
        return None

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

        # Only include ROIs with at least one spike
        if np.sum(spike_train) > 0:
            spike_trains[str(roi.id)] = spike_train

    return spike_trains if len(spike_trains) >= 2 else None


def _add_hover_functionality(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[str],
    synchrony_matrix: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[x], rois[y]
            sel.annotation.set(
                text=(
                    f"ROI {roi_x} ↔ ROI {roi_y}\n"
                    f"Spike Synchrony: {synchrony_matrix[y, x]:.3f}"
                ),
                fontsize=8,
                color="black",
            )
            if roi_x.isdigit() and roi_y.isdigit():
                widget.roiSelected.emit([roi_x, roi_y])
