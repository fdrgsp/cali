from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors
import numpy as np

from cali.plot._single_wells_plots._plot_calcium_peaks_correlation import (
    _calculate_cross_correlation,
)
from cali.plot._util import _create_connectivity_matrix

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from cali.gui._graph_widgets import _SingleWellGraphWidget

from cali.logger import cali_logger


def _plot_connectivity_network_data(
    widget: _SingleWellGraphWidget,
    db_path: str,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot spatial functional connectivity network.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    db_path : str
        Path to the database file
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    run_id : int | None
        The run ID to filter by, None for latest
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Calculate correlation matrix
    correlation_matrix, rois_idxs = _calculate_cross_correlation(
        db_path, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        cali_logger.warning(
            "Insufficient data for network connectivity analysis. "
            "Ensure at least two ROIs with calcium peaks are selected."
        )
        return

    # Get network threshold from AnalysisSettings
    network_threshold = _get_network_threshold(db_path, fov_name, run_id)

    # Create connectivity matrix
    connectivity_matrix = _create_connectivity_matrix(
        correlation_matrix, network_threshold
    )

    # Calculate and display network statistics
    n_nodes = len(rois_idxs)
    n_edges = (
        np.sum(connectivity_matrix) - n_nodes
    ) // 2  # Exclude diagonal, divide by 2 for symmetry
    total_possible_edges = n_nodes * (n_nodes - 1) // 2
    network_density = n_edges / total_possible_edges if total_possible_edges > 0 else 0

    # Display message about using detection viewer for spatial visualization
    ax.text(
        0.5,
        0.5,
        "Network Connectivity Visualization\n\n"
        f"Nodes: {n_nodes} ROIs\n"
        f"Edges: {n_edges}/{total_possible_edges}\n"
        f"Density: {network_density * 100:.1f}%\n"
        f"Threshold: {network_threshold:.1f}%\n\n"
        "For spatial network visualization,\n"
        "use the Detection Viewer with ROI masks.",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_network_threshold(
    db_path: str,
    fov_name: str,
    run_id: int | None = None,
) -> float:
    """Get network threshold from AnalysisSettings."""
    from sqlmodel import Session, col, create_engine, select

    from cali.sqlmodel._model import FOV, CaliResult, Experiment, Plate, Well

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
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
            return 90.0  # Default value

        threshold = result.analysis_settings.calcium_network_threshold
        return threshold if threshold is not None else 90.0


def _plot_connectivity_matrix_data(
    widget: _SingleWellGraphWidget,
    db_path: str,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot the binary connectivity matrix as a heatmap.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    db_path : str
        Path to the database file
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    run_id : int | None
        The run ID to filter by, None for latest
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Calculate correlation matrix
    correlation_matrix, rois_idxs = _calculate_cross_correlation(
        db_path, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        cali_logger.warning(
            "Insufficient data for connectivity matrix analysis. "
            "Ensure at least two ROIs with calcium peaks are selected."
        )
        return

    # Get network threshold
    network_threshold = _get_network_threshold(db_path, fov_name, run_id)

    # Create connectivity matrix
    connectivity_matrix = _create_connectivity_matrix(
        correlation_matrix, network_threshold
    )

    # Calculate network statistics
    n_nodes = len(rois_idxs)
    n_edges = np.sum(connectivity_matrix) - n_nodes  # Exclude diagonal
    total_possible_edges = n_nodes * (n_nodes - 1)
    network_density = n_edges / total_possible_edges if total_possible_edges > 0 else 0

    # Plot connectivity matrix
    img = ax.imshow(connectivity_matrix, cmap="binary", vmin=0, vmax=1)

    # Set labels and title
    ax.set_title(
        f"Binary Connectivity Matrix\n"
        f"Threshold: {network_threshold:.1f}% | "
        f"Edges: {n_edges // 2} | "  # Divide by 2 since matrix is symmetric
        f"Density: {network_density:.3f}",
        fontsize=12,
    )
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add hover functionality
    _add_hover_functionality_connectivity_matrix(
        img, widget, rois_idxs, connectivity_matrix, correlation_matrix
    )

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_connectivity_matrix(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    connectivity_matrix: np.ndarray,
    correlation_matrix: np.ndarray,
) -> None:
    """Add hover functionality to connectivity matrix heatmap."""
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[x], rois[y]
            is_connected = connectivity_matrix[y, x]
            correlation = correlation_matrix[y, x]

            status = "Connected" if is_connected else "Not Connected"

            sel.annotation.set(
                text=(
                    f"ROI {roi_x} ↔ ROI {roi_y}\n"
                    f"Status: {status}\n"
                    f"Correlation: {correlation:.3f}"
                ),
                fontsize=8,
                color="black",
            )
            widget.roiSelected.emit([str(roi_x), str(roi_y)])
