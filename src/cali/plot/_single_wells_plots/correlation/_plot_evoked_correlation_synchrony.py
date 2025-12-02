"""Evoked experiment correlation and synchrony plots.

Plots for stimulated vs non-stimulated ROIs. These wrappers filter ROIs
by stimulation status before calling the standard correlation and synchrony
plotting functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI

from ._plot_calcium_peaks_correlation import _plot_cross_correlation_data
from ._plot_calcium_peaks_synchrony import _plot_peak_event_synchrony_data
from ._plot_inferred_spike_correlation import _plot_spike_cross_correlation_data
from ._plot_inferred_spike_synchrony import _plot_spike_synchrony_data

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


def _filter_rois_by_stimulation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None,
    stimulated: bool,
) -> list[int] | None:
    """Filter ROIs by stimulation status.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        FOV name to query
    rois : list[int] | None
        Initial ROI filter (None for all ROIs in FOV)
    stimulated : bool
        If True, return only stimulated ROIs. If False, return only non-stimulated.

    Returns
    -------
    list[int] | None
        Filtered list of ROI label_values, or None if no ROIs match
    """
    with Session(engine) as session:
        stmt = (
            select(ROI.label_value)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.stimulated) == stimulated)  # Filter by stimulation status
            .where(col(ROI.active) == True)  # noqa: E712
        )

        # Apply user ROI filter if provided
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        filtered_rois = list(session.exec(stmt).all())

    return filtered_rois if filtered_rois else None


# =============================================================================
# Calcium Peaks - Stimulated ROIs
# =============================================================================


def _plot_stimulated_calcium_synchrony(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks synchrony for stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(engine, fov_name, rois, stimulated=True)
    _plot_peak_event_synchrony_data(widget, engine, fov_name, filtered_rois, run_id)


def _plot_stimulated_calcium_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks cross-correlation for stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(engine, fov_name, rois, stimulated=True)
    _plot_cross_correlation_data(widget, engine, fov_name, filtered_rois, run_id)


# =============================================================================
# Calcium Peaks - Non-Stimulated ROIs
# =============================================================================


def _plot_non_stimulated_calcium_synchrony(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks synchrony for non-stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(
        engine, fov_name, rois, stimulated=False
    )
    _plot_peak_event_synchrony_data(widget, engine, fov_name, filtered_rois, run_id)


def _plot_non_stimulated_calcium_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks cross-correlation for non-stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(
        engine, fov_name, rois, stimulated=False
    )
    _plot_cross_correlation_data(widget, engine, fov_name, filtered_rois, run_id)


# =============================================================================
# Inferred Spikes - Stimulated ROIs
# =============================================================================


def _plot_stimulated_spike_synchrony(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes synchrony for stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(engine, fov_name, rois, stimulated=True)
    _plot_spike_synchrony_data(widget, engine, fov_name, filtered_rois, run_id)


def _plot_stimulated_spike_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes cross-correlation for stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(engine, fov_name, rois, stimulated=True)
    _plot_spike_cross_correlation_data(widget, engine, fov_name, filtered_rois, run_id)


# =============================================================================
# Inferred Spikes - Non-Stimulated ROIs
# =============================================================================


def _plot_non_stimulated_spike_synchrony(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes synchrony for non-stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(
        engine, fov_name, rois, stimulated=False
    )
    _plot_spike_synchrony_data(widget, engine, fov_name, filtered_rois, run_id)


def _plot_non_stimulated_spike_correlation(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot inferred spikes cross-correlation for non-stimulated ROIs only."""
    filtered_rois = _filter_rois_by_stimulation(
        engine, fov_name, rois, stimulated=False
    )
    _plot_spike_cross_correlation_data(widget, engine, fov_name, filtered_rois, run_id)
