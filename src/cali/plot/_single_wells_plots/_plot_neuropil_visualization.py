"""Plot neuropil and ROI masks visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _plot_neuropil_masks(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot neuropil and ROI masks on widget canvas.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget containing the matplotlib figure and canvas
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of specific ROI IDs to plot, or None for all
    run_id : int | None
        The run ID to filter by, None for latest
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # TODO: Implement neuropil mask visualization from database
    ax.text(
        0.5,
        0.5,
        "Neuropil mask visualization\nnot yet implemented for database schema.\n\n"
        "Please use the detection viewer to view ROI and neuropil masks.",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.axis("off")
    widget.figure.tight_layout()
    widget.canvas.draw()
