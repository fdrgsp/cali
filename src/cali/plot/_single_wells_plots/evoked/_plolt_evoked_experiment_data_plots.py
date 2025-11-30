from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from skimage.measure import find_contours

from cali.plot._hover_utils import setup_pick_click

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"
P1 = 5
P2 = 100


# -----------------------------------------------------------------------------#
# Entry point dispatcher
# -----------------------------------------------------------------------------#
def _plot_evoked_experiment_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated_area: bool = False,
    with_rois: bool = False,
    with_peaks: bool = False,
) -> None:
    """Plot evoked experiment data.

    If `with_rois` or `stimulated_area` is True, show spatial maps.
    Otherwise, show stimulated vs non-stimulated dec ΔF/F traces.
    """
    if with_rois or stimulated_area:
        _visualize_stimulated_area(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=rois,
            run_id=run_id,
            with_rois=with_rois,
            stimulated_area=stimulated_area,
        )
    else:
        _plot_stimulated_vs_non_stimulated_roi_traces(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=rois,
            run_id=run_id,
            with_peaks=with_peaks,
        )


# -----------------------------------------------------------------------------#
# Peak amplitudes per ROI
# -----------------------------------------------------------------------------#
def _plot_stim_or_not_stim_peaks_amplitude(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated: bool = False,
) -> None:
    """Visualize stimulated peak amplitudes per ROI per stimulation parameters."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Query database for ROI data
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.stimulated) == stimulated)
        )

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            f"No {'stimulated' if stimulated else 'non-stimulated'} ROI data found.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    roi_labels: list[int] = []

    for roi_model, _traces, data_analysis in results:
        if data_analysis and data_analysis.peaks_amplitudes_dec_dff:
            roi_labels.append(roi_model.label_value)
            amps = np.asarray(data_analysis.peaks_amplitudes_dec_dff, dtype=float)

            mean_amp = float(np.mean(amps))
            if amps.size > 1:
                std_amp = float(np.std(amps, ddof=1))
                sem_amp = std_amp / np.sqrt(amps.size)
            else:
                sem_amp = 0.0

            label = f"ROI {roi_model.label_value}"
            # Mean ± SEM
            errorbar = ax.errorbar(
                [roi_model.label_value],
                [mean_amp],
                yerr=[sem_amp],
                fmt="o",
                capsize=5,
                color=STIMULATED_COLOR if stimulated else NON_STIMULATED_COLOR,
                label=label,
                zorder=2,
                picker=5,
            )
            if hasattr(errorbar, "lines") and errorbar.lines:
                errorbar.lines[0].set_picker(5)
                errorbar.lines[0].set_label(label)

            # Individual amplitudes
            ax.scatter(
                [roi_model.label_value] * amps.size,
                amps,
                alpha=0.5,
                s=30,
                color="lightgray",
                zorder=1,
                label=label,
                picker=True,
            )

    if not roi_labels:
        ax.text(
            0.5,
            0.5,
            "No peak amplitude data available.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    ax.set_xlabel("ROI")
    ax.set_ylabel("Peak Amplitude (dec ΔF/F)")
    title = "Stimulated" if stimulated else "Non-Stimulated"
    ax.set_title(f"{title} ROI Mean Peak Amplitudes ± SEM")
    ax.set_xticks(roi_labels)
    ax.set_xticklabels([])  # hide tick labels but keep positions

    # Disable coordinate display
    for ax in widget.figure.axes:
        ax.format_coord = lambda x, y: ""

    setup_pick_click(ax, widget, picker_tolerance=5)

    widget.figure.tight_layout()
    widget.canvas.draw()


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#
def extract_leading_number(key: str) -> float:
    """Extract leading number from key (before '_'), stripping units if present."""
    if match := re.match(r"(\d+(?:\.\d+)?)", key.split("_")[0]):
        return float(match[1])
    raise ValueError(f"Could not extract a valid number from key: {key}")


# -----------------------------------------------------------------------------#
# Spatial visualization: stimulation mask + ROI masks
# -----------------------------------------------------------------------------#
def _visualize_stimulated_area(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_rois: bool = False,
    stimulated_area: bool = False,
) -> None:
    """Visualize stimulated area with ROI mask overlay."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, AnalysisSettings, CaliResult, Mask
    from cali.util import coordinates_to_mask

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    stim_mask = None
    roi_data: list[tuple[ROI, Mask | None]] = []
    image_shape: tuple[int, int] | None = None

    with Session(engine) as session:
        session.expire_all()

        # Stimulation mask from AnalysisSettings
        result = session.get(CaliResult, run_id)
        if result and result.analysis_settings_id:
            analysis_settings = session.get(
                AnalysisSettings, result.analysis_settings_id
            )
            if analysis_settings and analysis_settings.stimulation_mask_id:
                mask_obj = session.get(Mask, analysis_settings.stimulation_mask_id)
                if (
                    mask_obj
                    and mask_obj.coords_y is not None
                    and mask_obj.coords_x is not None
                    and mask_obj.height is not None
                    and mask_obj.width is not None
                ):
                    coords = (mask_obj.coords_y, mask_obj.coords_x)
                    shape = (mask_obj.height, mask_obj.width)
                    stim_mask = coordinates_to_mask(coords, shape)
                    image_shape = shape

        # ROI + masks
        stmt = (
            select(ROI, Mask)
            .join(FOV, ROI.fov_id == FOV.id)
            .outerjoin(Mask, ROI.roi_mask_id == Mask.id)
            .where(col(FOV.name) == fov_name)
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()
        roi_data = [(roi, mask) for roi, mask in results]

        if image_shape is None and roi_data:
            for _roi, mask in roi_data:
                if mask and mask.height and mask.width:
                    image_shape = (mask.height, mask.width)
                    break

    if not roi_data:
        ax.text(
            0.5,
            0.5,
            "No ROI data found for visualization.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    stimulated_roi_labels: list[int] = []
    non_stimulated_roi_labels: list[int] = []
    for roi, _ in roi_data:
        if roi.stimulated:
            stimulated_roi_labels.append(roi.label_value)
        else:
            non_stimulated_roi_labels.append(roi.label_value)

    if with_rois and image_shape:
        labels = np.zeros(image_shape, dtype=np.int32)
        for roi, mask in roi_data:
            if (
                mask
                and mask.coords_y is not None
                and mask.coords_x is not None
                and mask.height is not None
                and mask.width is not None
            ):
                roi_mask = coordinates_to_mask(
                    (mask.coords_y, mask.coords_x), (mask.height, mask.width)
                )
                labels[roi_mask] = roi.label_value

        # color mapping
        color_mapping: dict[int, str] = {0: "black"}
        for roi, _ in roi_data:
            color_mapping[roi.label_value] = (
                STIMULATED_COLOR if roi.stimulated else NON_STIMULATED_COLOR
            )

        unique_labels = np.unique(labels)
        colors = [color_mapping.get(lbl, DEFAULT_COLOR) for lbl in unique_labels]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            boundaries=np.append(unique_labels, unique_labels[-1] + 1),
            ncolors=len(colors),
        )

        ax.imshow(labels, cmap=cmap, norm=norm)

        # stimulation area contours
        if stimulated_area and stim_mask is not None:
            stim_area_contours = find_contours(stim_mask.astype(float), level=0.5)
            for contour in stim_area_contours:
                ax.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=2)

        legend_patches = [
            Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
            Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
        ]
        if stimulated_area and stim_mask is not None:
            legend_patches.append(Patch(color="yellow", label="Stimulation Area"))

        ax.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_patches),
            frameon=True,
            fontsize="small",
            edgecolor="black",
        )
    elif stimulated_area and stim_mask is not None:
        ax.imshow(stim_mask, cmap="gray", clim=(0, 1))
    else:
        _display_roi_statistics(ax, stimulated_roi_labels, non_stimulated_roi_labels)

    ax.axis("off")
    widget.figure.tight_layout()
    widget.canvas.draw()


def _display_roi_statistics(
    ax: Axes,
    stimulated_rois: list[int],
    non_stimulated_rois: list[int],
) -> None:
    """Display ROI statistics as text when mask visualization is not available."""
    text_lines = [
        "Stimulated Area Visualization",
        "",
        f"Total ROIs: {len(stimulated_rois) + len(non_stimulated_rois)}",
        f"Stimulated ROIs: {len(stimulated_rois)}",
        f"Non-Stimulated ROIs: {len(non_stimulated_rois)}",
        "",
    ]
    ax.text(
        0.5,
        0.5,
        "\n".join(text_lines),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
        family="monospace",
    )


# -----------------------------------------------------------------------------#
# dec ΔF/F traces: stimulated vs non-stimulated
# -----------------------------------------------------------------------------#
def _plot_stimulated_vs_non_stimulated_roi_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_peaks: bool = False,
) -> None:
    """Plot dec ΔF/F traces with global percentile normalization (5th-100th)."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            "No ROI data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    stimulated_data: list[tuple] = []
    non_stimulated_data: list[tuple] = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.dec_dff:
            if roi_model.stimulated:
                stimulated_data.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_data.append((roi_model, trace_obj, data_analysis))

            if data_analysis and data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    # Global percentile normalization
    all_values: list[float] = []
    for _, trace_obj, _ in results:
        if trace_obj and trace_obj.dec_dff:
            all_values.extend(trace_obj.dec_dff)

    if all_values:
        percentiles = np.percentile(all_values, [P1, P2])
        p1, p2 = float(percentiles[0]), float(percentiles[1])
    else:
        p1, p2 = 0.0, 1.0

    count = 0
    last_trace: list[float] | None = None

    # Stimulated traces (vectorized per-group)
    stim_norm_traces: list[np.ndarray] = []
    for _roi_model, trace_obj, _data_analysis in stimulated_data:
        if trace_obj.dec_dff:
            trace = np.asarray(
                _normalize_trace_percentile(trace_obj.dec_dff, p1, p2), dtype=float
            )
            stim_norm_traces.append(trace)
            last_trace = trace_obj.dec_dff

    if stim_norm_traces:
        Y = np.vstack(stim_norm_traces)
        n_stim = Y.shape[0]
        offsets = (np.arange(n_stim) * 1.1)[:, None]
        ax.plot((Y + offsets).T, color=STIMULATED_COLOR, linewidth=1)
        count += n_stim

    # Peaks for stimulated (still per-ROI)
    if with_peaks:
        for idx, (_, trace_obj, data_analysis) in enumerate(stimulated_data):
            if not (
                trace_obj.dec_dff and data_analysis and data_analysis.peaks_dec_dff
            ):
                continue
            trace = np.asarray(
                _normalize_trace_percentile(trace_obj.dec_dff, p1, p2), dtype=float
            )
            offset = idx * 1.1
            peaks_indices = [int(p) for p in data_analysis.peaks_dec_dff]
            ax.plot(
                peaks_indices,
                trace[peaks_indices] + offset,
                "x",
                color="black",
                markersize=8,
            )

    # Non-stimulated traces
    non_stim_norm_traces: list[np.ndarray] = []
    for _roi_model, trace_obj, _data_analysis in non_stimulated_data:
        if trace_obj.dec_dff:
            trace = np.asarray(
                _normalize_trace_percentile(trace_obj.dec_dff, p1, p2), dtype=float
            )
            non_stim_norm_traces.append(trace)
            last_trace = trace_obj.dec_dff

    if non_stim_norm_traces:
        Y = np.vstack(non_stim_norm_traces)
        n_non = Y.shape[0]
        offsets = (np.arange(count, count + n_non) * 1.1)[:, None]
        ax.plot((Y + offsets).T, color=NON_STIMULATED_COLOR, linewidth=1)
        count += n_non

    if with_peaks:
        for idx, (_, trace_obj, data_analysis) in enumerate(non_stimulated_data):
            if not (
                trace_obj.dec_dff and data_analysis and data_analysis.peaks_dec_dff
            ):
                continue
            trace = np.asarray(
                _normalize_trace_percentile(trace_obj.dec_dff, p1, p2), dtype=float
            )
            offset = (idx + len(stimulated_data)) * 1.1
            peaks_indices = [int(p) for p in data_analysis.peaks_dec_dff]
            ax.plot(
                peaks_indices,
                trace[peaks_indices] + offset,
                "x",
                color="black",
                markersize=8,
            )

    ax.set_ylabel("Normalized dec ΔF/F")
    ax.set_title("Stimulated vs Non-Stimulated ROIs (Normalized)")
    ax.set_yticks([])
    ax.set_yticklabels([])

    legend_patches = [
        Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
        Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize="small")

    _update_time_axis(ax, rois_rec_time, last_trace)

    _add_hover_functionality_stim_vs_non_stim(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _normalize_trace_percentile(
    trace: list[float], p1: float, p2: float
) -> list[float]:
    """Normalize a trace using the global p1-p2 percentiles."""
    tr = np.array(trace, dtype=float)
    denom = p2 - p1
    if denom == 0:
        return cast("list[float]", np.zeros_like(tr).tolist())
    normalized = (tr - p1) / denom
    normalized = np.clip(normalized, 0, 1)
    return cast("list[float]", normalized.tolist())


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    avg_rec_time = int(np.mean(rois_rec_time))
    total_frames = len(trace) if trace is not None else 1
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")


def _add_hover_functionality_stim_vs_non_stim(
    ax: Axes, widget: _SingleWellGraphWidget
) -> None:
    setup_pick_click(ax, widget, picker_tolerance=5)


# -----------------------------------------------------------------------------#
# Spike raster: stimulated vs non-stimulated
# -----------------------------------------------------------------------------#
def _plot_stimulated_vs_non_stimulated_spike_raster(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot raster of thresholded spikes: green=stim, magenta=non-stim."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            "No active ROI data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    stimulated_rois: list[tuple] = []
    non_stimulated_rois: list[tuple] = []
    active_roi_labels: list[int] = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.inferred_spikes and data_analysis:
            active_roi_labels.append(roi_model.label_value)
            if roi_model.stimulated:
                stimulated_rois.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_rois.append((roi_model, trace_obj, data_analysis))

            if data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not stimulated_rois and not non_stimulated_rois:
        ax.text(
            0.5,
            0.5,
            "No spike data available.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    event_data: list[list[float]] = []
    colors: list[str] = []
    last_trace: list[float] | None = None

    # stim
    for _roi_model, trace_obj, data_analysis in stimulated_rois:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        spike_indices = np.where(spikes > 0.0)[0]
        event_data.append(spike_indices.tolist())
        colors.append(STIMULATED_COLOR)
        last_trace = trace_obj.inferred_spikes

    # non-stim
    for _roi_model, trace_obj, data_analysis in non_stimulated_rois:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        spike_indices = np.where(spikes > 0.0)[0]
        event_data.append(spike_indices.tolist())
        colors.append(NON_STIMULATED_COLOR)
        last_trace = trace_obj.inferred_spikes

    ax.eventplot(event_data, colors=colors, linewidth=2)

    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title("Stimulated vs Non-Stimulated Spike Raster Plot")

    legend_patches = [
        Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
        Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize="small")

    _update_time_axis_spike_traces(ax, rois_rec_time, last_trace)

    _add_hover_functionality_spike_traces(ax, widget, active_roi_labels)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_spike_traces(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    from cali.plot._hover_utils import setup_pick_click_for_raster

    setup_pick_click_for_raster(ax, widget, active_rois, picker_tolerance=5)


# -----------------------------------------------------------------------------#
# Spike traces: stimulated vs non-stimulated
# -----------------------------------------------------------------------------#
def _plot_stimulated_vs_non_stimulated_spike_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot continuous inferred spike traces separated by stimulation status."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            "No active ROI data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    stimulated_data: list[tuple] = []
    non_stimulated_data: list[tuple] = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.inferred_spikes:
            if roi_model.stimulated:
                stimulated_data.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_data.append((roi_model, trace_obj, data_analysis))

            if data_analysis and data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not stimulated_data and not non_stimulated_data:
        ax.text(
            0.5,
            0.5,
            "No spike data available.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    count = 0
    last_trace: list[float] | None = None

    # Stim traces
    for roi_model, trace_obj, data_analysis in stimulated_data:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis and data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        offset = count * 1.1
        ax.plot(
            spikes + offset,
            label=f"ROI {roi_model.label_value}",
            color=STIMULATED_COLOR,
            linewidth=1.5,
        )
        last_trace = trace_obj.inferred_spikes
        count += 1

    # Non-stim traces
    for roi_model, trace_obj, data_analysis in non_stimulated_data:
        spikes = np.asarray(trace_obj.inferred_spikes, dtype=float)
        if data_analysis and data_analysis.inferred_spikes_threshold is not None:
            the = float(data_analysis.inferred_spikes_threshold)
            spikes = np.where(spikes > the, spikes, 0.0)
        offset = count * 1.1
        ax.plot(
            spikes + offset,
            label=f"ROI {roi_model.label_value}",
            color=NON_STIMULATED_COLOR,
            linewidth=1.5,
        )
        last_trace = trace_obj.inferred_spikes
        count += 1

    ax.set_ylabel("Inferred Spikes (Thresholded)")
    ax.set_title(
        "Stimulated vs Non-Stimulated Spike Traces\n(Thresholded Inferred Spikes)"
    )
    ax.set_yticks([])
    ax.set_yticklabels([])

    legend_patches = [
        Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
        Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize="small")

    _update_time_axis_spike_traces(ax, rois_rec_time, last_trace)

    active_roi_labels = [
        roi_model.label_value
        for roi_model, _, _ in stimulated_data + non_stimulated_data
    ]
    _add_hover_functionality_spike_traces(ax, widget, active_roi_labels)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _update_time_axis_spike_traces(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Update x-axis to show time (s) if recording time is available."""
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    avg_rec_time = int(np.mean(rois_rec_time))
    total_frames = len(trace) if trace is not None else 1
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")
